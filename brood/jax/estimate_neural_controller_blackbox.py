#!/usr/bin/env python
"""
Brood baseline (Approach A++): Black-box surrogate controller with memory + delta-u parameterization.

Key ideas:
- Input includes u_prev (4 dims) so the policy has minimal memory.
- Network outputs dU (delta action), not absolute U.
- Applied control is: U = clamp(U_prev + dU)

Training:
- Phase 1: imitation on all samples (x_t, u_{t-1}) -> u_t
- Phase 2: differentiable rollout loss averaged across episodes:
    L = tracking + lambda_u * action_imitation + lambda_du * delta_action_penalty + lambda_tilt * tilt_penalty

Why this helps:
- Delta-u makes the controller stable-by-construction and reduces error accumulation in long rollouts.
- lambda_u keeps rollout updates from drifting away from teacher behavior.

Run:
  python estimate_neural_controller_blackbox.py \
    --logs_dir /path/to/logs \
    --runconfigs_dir /path/to/runconfigs \
    --epochs_imitation 15 --epochs_rollout 40 \
    --rollout_lr 3e-5 --clip_norm 0.5 \
    --lambda_u 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np

import jax
import jax.numpy as jnp
import optax
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------

def wrap_pi(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def rpy_to_rotmat(roll: jnp.ndarray, pitch: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)"""
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)

    Rz = jnp.array([[cy, -sy, 0.0],
                    [sy,  cy, 0.0],
                    [0.0, 0.0, 1.0]], dtype=jnp.float32)
    Ry = jnp.array([[ cp, 0.0, sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0, cp]], dtype=jnp.float32)
    Rx = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, cr, -sr],
                    [0.0, sr,  cr]], dtype=jnp.float32)
    return Rz @ Ry @ Rx


def safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = x.std(axis=0)
    s[s < eps] = 1.0
    return s


def parse_run_id_from_log_name(csv_name: str) -> str:
    """
    Expected: ep_something_YYYYMMDD_HHMMSS.csv
    Return:   ep_something
    """
    stem = Path(csv_name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        maybe_date = parts[-2]
        maybe_time = parts[-1]
        if maybe_date.isdigit() and len(maybe_date) == 8 and maybe_time.isdigit() and len(maybe_time) == 6:
            return "_".join(parts[:-2])
    return stem


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Episode:
    csv_path: Path
    run_id: str
    runconfig_path: Optional[Path]
    dt: float

    state0: np.ndarray      # (10,)
    ref_seq: np.ndarray     # (T,4)
    teacher_u: np.ndarray   # (T,4)
    X_raw: np.ndarray       # (T,13) = state(9)+ref(4)

    env_np: Dict[str, Any]


@dataclass(frozen=True)
class MultiDataset:
    state0s: jnp.ndarray        # (E,10)
    ref_seqs: jnp.ndarray       # (E,Tr,4)
    teacher_us: jnp.ndarray     # (E,Tr,4)

    # Input normalizers for x17 = [state9, ref4, u_prev4]
    x_mean: jnp.ndarray         # (17,)
    x_std: jnp.ndarray          # (17,)

    # Output normalizers for dU (delta-u)
    du_mean: jnp.ndarray        # (4,)
    du_std: jnp.ndarray         # (4,)

    # Also keep absolute U normalizers for saving / inspection
    u_mean: jnp.ndarray         # (4,)
    u_std: jnp.ndarray          # (4,)

    # Imitation arrays
    X_all_raw: jnp.ndarray      # (N,17)
    U_all: jnp.ndarray          # (N,4)
    U_prev_all: jnp.ndarray     # (N,4)
    dU_all: jnp.ndarray         # (N,4)
    ep_id_all: jnp.ndarray      # (N,)

    env: Dict[str, jnp.ndarray] # (E,) or (E,3)

    dt: float
    Tr: int
    E: int


# ----------------------------
# CSV loading
# ----------------------------

def load_csv_log(csv_path: Path) -> Dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)

    req = [
        "t",
        "px", "py", "pz",
        "vx", "vy", "vz",
        "roll", "pitch", "yaw",
        "prefx", "prefy", "prefz",
        "yaw_ref",
        "roll_cmd", "pitch_cmd", "yaw_rate_cmd", "thrust_cmd",
        "mass", "drag", "g",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing}")

    return {c: df[c].to_numpy(dtype=np.float32) for c in req}


# ----------------------------
# RunConfig reading (only estimator-relevant)
# ----------------------------

def read_env_from_runconfig(runconfig_path: Optional[Path]) -> Dict[str, Any]:
    env = {
        "tau_roll": 0.08,
        "tau_pitch": 0.08,
        "tau_thrust": 0.05,

        "wind": np.array([0.0, 0.0, 0.0], dtype=np.float32),

        "tilt_max": float(25.0 * math.pi / 180.0),
        "yaw_rate_max": float(180.0 * math.pi / 180.0),
        "thrust_min": 0.0,
        "thrust_max": 30.0,
    }

    if runconfig_path is None or not runconfig_path.exists():
        return env

    cfg = json.loads(runconfig_path.read_text(encoding="utf-8"))
    dyn = cfg.get("dynamics", {})
    ctrl = cfg.get("controller", {})

    if isinstance(dyn, dict):
        env["tau_roll"] = float(dyn.get("tau_roll", env["tau_roll"]))
        env["tau_pitch"] = float(dyn.get("tau_pitch", env["tau_pitch"]))
        env["tau_thrust"] = float(dyn.get("tau_thrust", env["tau_thrust"]))

        wind = dyn.get("wind", None)
        if isinstance(wind, dict) and wind.get("type") == "constant":
            vec = wind.get("vec", [0.0, 0.0, 0.0])
            env["wind"] = np.array(vec, dtype=np.float32)

    if isinstance(ctrl, dict):
        env["tilt_max"] = float(float(ctrl.get("tilt_max_deg", 25.0)) * math.pi / 180.0)
        env["yaw_rate_max"] = float(float(ctrl.get("yaw_rate_max_dps", 180.0)) * math.pi / 180.0)
        env["thrust_min"] = float(ctrl.get("thrust_min", 0.0))
        env["thrust_max"] = float(ctrl.get("thrust_max", 30.0))

    return env


def load_episode(csv_path: Path, runconfig_path: Optional[Path]) -> Episode:
    log = load_csv_log(csv_path)
    T = log["t"].shape[0]
    dt = float(log["t"][1] - log["t"][0]) if T >= 2 else 0.01

    run_id = parse_run_id_from_log_name(csv_path.name)

    thrust0 = float(log["thrust_cmd"][0])
    state0 = np.array([
        log["px"][0], log["py"][0], log["pz"][0],
        log["vx"][0], log["vy"][0], log["vz"][0],
        log["roll"][0], log["pitch"][0], log["yaw"][0],
        thrust0,
    ], dtype=np.float32)

    ref_seq = np.stack([log["prefx"], log["prefy"], log["prefz"], log["yaw_ref"]], axis=1).astype(np.float32)
    teacher_u = np.stack([log["roll_cmd"], log["pitch_cmd"], log["yaw_rate_cmd"], log["thrust_cmd"]], axis=1).astype(np.float32)

    state_seq = np.stack([
        log["px"], log["py"], log["pz"],
        log["vx"], log["vy"], log["vz"],
        log["roll"], log["pitch"], log["yaw"],
    ], axis=1).astype(np.float32)

    X_raw = np.concatenate([state_seq, ref_seq], axis=1).astype(np.float32)  # (T,13)

    env_np = read_env_from_runconfig(runconfig_path)
    env_np["mass"] = float(np.mean(log["mass"]))
    env_np["drag"] = float(np.mean(log["drag"]))
    env_np["g"] = float(np.mean(log["g"]))

    return Episode(
        csv_path=csv_path,
        run_id=run_id,
        runconfig_path=runconfig_path if (runconfig_path and runconfig_path.exists()) else None,
        dt=dt,
        state0=state0,
        ref_seq=ref_seq,
        teacher_u=teacher_u,
        X_raw=X_raw,
        env_np=env_np,
    )


def build_multi_dataset(episodes: List[Episode]) -> MultiDataset:
    if not episodes:
        raise ValueError("No episodes provided")

    Tr = int(min(ep.ref_seq.shape[0] for ep in episodes))
    E = len(episodes)

    dt0 = episodes[0].dt
    for ep in episodes[1:]:
        if abs(ep.dt - dt0) > 1e-6:
            print(f"[warn] dt mismatch: {ep.csv_path.name} dt={ep.dt} vs dt0={dt0} (using dt0)")

    X_list, U_list, Uprev_list, dU_list, I_list = [], [], [], [], []

    for i, ep in enumerate(episodes):
        U = ep.teacher_u.astype(np.float32)  # (T,4)
        X13 = ep.X_raw.astype(np.float32)    # (T,13)

        U_prev = np.vstack([U[0:1], U[:-1]]).astype(np.float32)   # (T,4)
        dU = (U - U_prev).astype(np.float32)                      # (T,4)

        X17 = np.concatenate([X13, U_prev], axis=1).astype(np.float32)  # (T,17)

        X_list.append(X17)
        U_list.append(U)
        Uprev_list.append(U_prev)
        dU_list.append(dU)
        I_list.append(np.full((X17.shape[0],), i, dtype=np.int32))

    X_all = np.concatenate(X_list, axis=0)            # (N,17)
    U_all = np.concatenate(U_list, axis=0)            # (N,4)
    U_prev_all = np.concatenate(Uprev_list, axis=0)   # (N,4)
    dU_all = np.concatenate(dU_list, axis=0)          # (N,4)
    ep_id_all = np.concatenate(I_list, axis=0)        # (N,)

    # Normalizers
    x_mean = X_all.mean(axis=0)
    x_std = safe_std(X_all)

    u_mean = U_all.mean(axis=0)
    u_std = safe_std(U_all)

    du_mean = dU_all.mean(axis=0)
    du_std = safe_std(dU_all)

    # Episode-major (cropped) arrays for rollout training/eval
    state0s = np.stack([ep.state0 for ep in episodes], axis=0).astype(np.float32)
    ref_seqs = np.stack([ep.ref_seq[:Tr] for ep in episodes], axis=0).astype(np.float32)
    teacher_us = np.stack([ep.teacher_u[:Tr] for ep in episodes], axis=0).astype(np.float32)

    # Env per episode
    env_keys = episodes[0].env_np.keys()
    env: Dict[str, np.ndarray] = {}
    for k in env_keys:
        vals = [ep.env_np[k] for ep in episodes]
        if isinstance(vals[0], np.ndarray):
            env[k] = np.stack(vals, axis=0).astype(np.float32)
        else:
            env[k] = np.array(vals, dtype=np.float32)

    return MultiDataset(
        state0s=jnp.asarray(state0s),
        ref_seqs=jnp.asarray(ref_seqs),
        teacher_us=jnp.asarray(teacher_us),
        x_mean=jnp.asarray(x_mean),
        x_std=jnp.asarray(x_std),
        du_mean=jnp.asarray(du_mean),
        du_std=jnp.asarray(du_std),
        u_mean=jnp.asarray(u_mean),
        u_std=jnp.asarray(u_std),
        X_all_raw=jnp.asarray(X_all),
        U_all=jnp.asarray(U_all),
        U_prev_all=jnp.asarray(U_prev_all),
        dU_all=jnp.asarray(dU_all),
        ep_id_all=jnp.asarray(ep_id_all),
        env={k: jnp.asarray(v) for k, v in env.items()},
        dt=dt0,
        Tr=Tr,
        E=E,
    )


# ----------------------------
# MLP predicts dU
# ----------------------------

def init_mlp_params(key: jax.Array, in_dim: int, hidden: int, depth: int, out_dim: int = 4) -> list:
    keys = jax.random.split(key, depth + 1)
    params = []
    d = in_dim
    for i in range(depth):
        W = jax.random.normal(keys[i], (d, hidden), dtype=jnp.float32) * jnp.sqrt(2.0 / d)
        b = jnp.zeros((hidden,), dtype=jnp.float32)
        params.append((W, b))
        d = hidden
    W = jax.random.normal(keys[-1], (d, out_dim), dtype=jnp.float32) * jnp.sqrt(2.0 / d)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    params.append((W, b))
    return params


def mlp_apply(params: list, x: jax.Array) -> jax.Array:
    for (W, b) in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def clamp_u(u: jax.Array, env_ep: Dict[str, jax.Array]) -> jax.Array:
    tilt_max = env_ep["tilt_max"]
    yaw_rate_max = env_ep["yaw_rate_max"]
    thrust_min = env_ep["thrust_min"]
    thrust_max = env_ep["thrust_max"]

    roll = jnp.clip(u[0], -tilt_max, tilt_max)
    pitch = jnp.clip(u[1], -tilt_max, tilt_max)
    yaw_rate = jnp.clip(u[2], -yaw_rate_max, yaw_rate_max)
    thrust = jnp.clip(u[3], thrust_min, thrust_max)

    return jnp.array([roll, pitch, yaw_rate, thrust], dtype=jnp.float32)


def controller_delta_u(
    params: list,
    state: jax.Array,               # (10,)
    ref_t: jax.Array,               # (4,)
    u_prev: jax.Array,              # (4,)
    env_ep: Dict[str, jax.Array],
    x_mean: jax.Array,
    x_std: jax.Array,
    du_mean: jax.Array,
    du_std: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Returns (u, dU) where:
      dU is the *physical* delta action (not normalized)
      u = clamp(u_prev + dU)
    """
    x_raw13 = jnp.concatenate([state[0:9], ref_t])
    x_raw17 = jnp.concatenate([x_raw13, u_prev])
    x = (x_raw17 - x_mean) / x_std

    dU_norm = mlp_apply(params, x)
    dU = dU_norm * du_std + du_mean
    u = clamp_u(u_prev + dU, env_ep)
    return u, dU


# ----------------------------
# Dynamics
# ----------------------------

def dynamics_step(state: jax.Array, u: jax.Array, env_ep: Dict[str, jax.Array], dt: float) -> jax.Array:
    p = state[0:3]
    v = state[3:6]
    roll, pitch, yaw = state[6], state[7], state[8]
    thrust_actual = state[9]

    roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd = u

    mass = env_ep["mass"]
    drag = env_ep["drag"]
    g = env_ep["g"]
    tau_roll = env_ep["tau_roll"]
    tau_pitch = env_ep["tau_pitch"]
    tau_thrust = env_ep["tau_thrust"]
    wind = env_ep["wind"]

    roll = roll + (dt / jnp.maximum(tau_roll, 1e-6)) * (roll_cmd - roll)
    pitch = pitch + (dt / jnp.maximum(tau_pitch, 1e-6)) * (pitch_cmd - pitch)
    yaw = wrap_pi(yaw + yaw_rate_cmd * dt)

    thrust_actual = thrust_actual + (dt / jnp.maximum(tau_thrust, 1e-6)) * (thrust_cmd - thrust_actual)
    T = thrust_actual

    R = rpy_to_rotmat(roll, pitch, yaw)
    b3 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    F = T * (R @ b3)

    gvec = jnp.array([0.0, 0.0, -g], dtype=jnp.float32)
    a = (F / jnp.maximum(mass, 1e-6)) + gvec - drag * v + wind

    v = v + a * dt
    p = p + v * dt

    return jnp.concatenate([p, v, jnp.array([roll, pitch, yaw, thrust_actual], dtype=jnp.float32)])


# ----------------------------
# Rollouts
# ----------------------------

def rollout_one_delta(
    params: list,
    state0: jax.Array,              # (10,)
    ref_seq: jax.Array,             # (Tr,4)
    env_ep: Dict[str, jax.Array],
    dt: float,
    x_mean: jax.Array,
    x_std: jax.Array,
    du_mean: jax.Array,
    du_std: jax.Array,
    u0: jax.Array,                  # (4,) initial u_prev
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns:
      states: (Tr,10) states before applying control at each step
      us:     (Tr,4)  control applied
      dUs:    (Tr,4)  delta applied (physical units)
    """
    def step_fn(carry, ref_t):
        state, u_prev = carry
        u, dU = controller_delta_u(params, state, ref_t, u_prev, env_ep, x_mean, x_std, du_mean, du_std)
        nxt = dynamics_step(state, u, env_ep, dt)
        return (nxt, u), (state, u, dU)

    (_, _), (states, us, dUs) = jax.lax.scan(step_fn, (state0, u0), ref_seq)
    return states, us, dUs

def rollout_teacher_in_our_plant_with_u(
    state0: jax.Array,
    u_seq: jax.Array,               # (Tr,4)
    env_ep: Dict[str, jax.Array],
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Same as rollout_teacher_in_our_plant, but also returns the (clamped) actions actually applied.
    Returns:
      states: (Tr,10) states before each step
      us:     (Tr,4)  clamped teacher actions
    """
    def step_fn(state, u):
        u_clamped = clamp_u(u, env_ep)
        nxt = dynamics_step(state, u_clamped, env_ep, dt)
        return nxt, (state, u_clamped)

    _, (states, us) = jax.lax.scan(step_fn, state0, u_seq)
    return states, us

def rollout_teacher_in_our_plant(
    state0: jax.Array,
    u_seq: jax.Array,               # (Tr,4)
    env_ep: Dict[str, jax.Array],
    dt: float,
) -> jax.Array:
    """
    Roll forward our plant using teacher commands (clamped for fairness).
    Returns states (Tr,10) before each step.
    """
    def step_fn(state, u):
        u = clamp_u(u, env_ep)
        nxt = dynamics_step(state, u, env_ep, dt)
        return nxt, state

    _, states = jax.lax.scan(step_fn, state0, u_seq)
    return states


# ----------------------------
# Losses + train steps
# ----------------------------

def one_step_imitation_loss_delta(
    params: list,
    Xb_raw17: jax.Array,            # (B,17) [state9, ref4, u_prev4]
    Ub_teacher: jax.Array,          # (B,4)  u_t
    ep_idx: jax.Array,              # (B,)
    mds: MultiDataset,
) -> jax.Array:
    env_keys = list(mds.env.keys())

    def pred_u(x17, u_teacher, i):
        i = i.astype(jnp.int32)
        env_ep = {k: mds.env[k][i] for k in env_keys}

        state9 = x17[0:9]
        ref_t = x17[9:13]
        u_prev = x17[13:17]

        # thrust_actual proxy for one-step: use teacher thrust at that step (consistent with earlier script)
        thrust_actual = u_teacher[3]
        state = jnp.concatenate([state9, jnp.array([thrust_actual], dtype=jnp.float32)])

        u_hat, _ = controller_delta_u(params, state, ref_t, u_prev, env_ep,
                                      mds.x_mean, mds.x_std, mds.du_mean, mds.du_std)
        return u_hat

    u_hat = jax.vmap(pred_u)(Xb_raw17, Ub_teacher, ep_idx)
    return jnp.mean((u_hat - Ub_teacher) ** 2)


def rollout_loss_delta(
    params: list,
    mds: MultiDataset,
    lambda_u: float,
    lambda_du: float,
    lambda_tilt: float,
) -> jax.Array:
    env_keys = list(mds.env.keys())

    def loss_one(i: jax.Array) -> jax.Array:
        i = i.astype(jnp.int32)
        env_ep = {k: mds.env[k][i] for k in env_keys}

        u_teacher = mds.teacher_us[i]          # (Tr,4)
        ref_seq = mds.ref_seqs[i]              # (Tr,4)

        # initial u_prev = teacher u0 (good anchor)
        u0 = clamp_u(u_teacher[0], env_ep)

        states, us, dUs = rollout_one_delta(
            params,
            mds.state0s[i],
            ref_seq,
            env_ep,
            mds.dt,
            mds.x_mean, mds.x_std,
            mds.du_mean, mds.du_std,
            u0=u0
        )

        p = states[:, 0:3]
        p_ref = ref_seq[:, 0:3]
        track = jnp.mean(jnp.sum((p - p_ref) ** 2, axis=-1))

        # keep actions close to teacher during rollout (reduces drifting policies)
        imit_u = jnp.mean(jnp.sum((us - u_teacher) ** 2, axis=-1))

        # penalize large delta actions (stability / smoothness)
        du_pen = jnp.mean(jnp.sum(dUs ** 2, axis=-1))

        tilt = jnp.mean(jnp.sum(states[:, 6:8] ** 2, axis=-1))

        return track + lambda_u * imit_u + lambda_du * du_pen + lambda_tilt * tilt

    losses = jax.vmap(loss_one)(jnp.arange(mds.E))
    return jnp.mean(losses)


def make_train_step_imitation(optimizer, mds: MultiDataset):
    @jax.jit
    def step(params, opt_state, Xb, Ub, Ib):
        loss, grads = jax.value_and_grad(one_step_imitation_loss_delta)(params, Xb, Ub, Ib, mds)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def make_train_step_rollout(optimizer, mds: MultiDataset, lambda_u: float, lambda_du: float, lambda_tilt: float):
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(rollout_loss_delta)(params, mds, lambda_u, lambda_du, lambda_tilt)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


# ----------------------------
# Main
# ----------------------------

def minibatches(rng: np.random.Generator, X: np.ndarray, U: np.ndarray, ep_idx: np.ndarray, batch_size: int):
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for i in range(0, N, batch_size):
        j = idx[i:i + batch_size]
        if j.shape[0] != batch_size:
            continue
        yield X[j], U[j], ep_idx[j]

def write_rollout_csv(path, t, states, us, label):
    """
    states: [T, state_dim]
    us:     [T, action_dim]
    """
    import csv

    header = (
        ["t", "label"] +
        [f"x{i}" for i in range(states.shape[1])] +
        [f"u{i}" for i in range(us.shape[1])]
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t)):
            writer.writerow(
                [float(t[i]), label] +
                list(map(float, states[i])) +
                list(map(float, us[i]))
            )

def plot_xyz(path, t, p_teacher, p_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["x", "y", "z"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axs[i].plot(t, p_teacher[:, i], label="teacher", linewidth=2)
        axs[i].plot(t, p_model[:, i], label="surrogate", linestyle="--")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        if i == 0:
            axs[i].legend()

    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def write_diag_csv_combined(
    path: str,
    t: np.ndarray,
    ref_seq: np.ndarray,         # (T,4) [prefx,prefy,prefz,yaw_ref]
    states_teacher: np.ndarray,  # (T,10)
    u_teacher: np.ndarray,       # (T,4)
    states_sur: np.ndarray,      # (T,10)
    u_sur: np.ndarray,           # (T,4)
    dU_sur: np.ndarray,          # (T,4)
):
    """
    Writes a single CSV with teacher + surrogate + reference for direct numerical comparison.
    Schema is explicit (no x0/x1 guessing).
    """
    import csv

    header = [
        "t",

        # reference
        "prefx", "prefy", "prefz", "yaw_ref",

        # teacher state
        "px_T", "py_T", "pz_T",
        "vx_T", "vy_T", "vz_T",
        "roll_T", "pitch_T", "yaw_T",
        "thrust_actual_T",

        # teacher action (after clamp)
        "roll_cmd_T", "pitch_cmd_T", "yaw_rate_cmd_T", "thrust_cmd_T",

        # surrogate state
        "px_S", "py_S", "pz_S",
        "vx_S", "vy_S", "vz_S",
        "roll_S", "pitch_S", "yaw_S",
        "thrust_actual_S",

        # surrogate action (applied, clamped)
        "roll_cmd_S", "pitch_cmd_S", "yaw_rate_cmd_S", "thrust_cmd_S",

        # surrogate delta-u (physical units)
        "droll_S", "dpitch_S", "dyawrate_S", "dthrust_S",
    ]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        T = len(t)
        for k in range(T):
            row = (
                [float(t[k])] +
                list(map(float, ref_seq[k])) +
                list(map(float, states_teacher[k])) +
                list(map(float, u_teacher[k])) +
                list(map(float, states_sur[k])) +
                list(map(float, u_sur[k])) +
                list(map(float, dU_sur[k]))
            )
            w.writerow(row)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=str, default=None, help="Directory containing *.csv logs (preferred)")
    ap.add_argument("--runconfigs_dir", type=str, default=None, help="Directory containing matching *.json runconfigs")
    ap.add_argument("--controller", type=str, default=None, help="Legacy: use ../logs/<controller> and ../runconfigs/<controller>")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--lr", type=float, default=1e-3, help="imitation LR")
    ap.add_argument("--rollout_lr", type=float, default=3e-5, help="rollout LR")
    ap.add_argument("--clip_norm", type=float, default=0.5)

    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs_imitation", type=int, default=15)
    ap.add_argument("--epochs_rollout", type=int, default=40)

    ap.add_argument("--lambda_u", type=float, default=1.0, help="rollout action imitation weight")
    ap.add_argument("--lambda_du", type=float, default=1e-3, help="rollout delta-action penalty weight")
    ap.add_argument("--lambda_tilt", type=float, default=1e-3, help="rollout tilt penalty weight")
    ap.add_argument("--diag_episode", type=int, default=0,help="Which episode index to export diagnostics for")
    ap.add_argument("--diag_out", type=str, default="diagnostics",help="Output directory for rollout diagnostics")
    ap.add_argument("--save", type=str, default="brood_baseline_A_deltau_params.npz")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.logs_dir and args.runconfigs_dir:
        logs_dir = Path(args.logs_dir).expanduser().resolve()
        runconfigs_dir = Path(args.runconfigs_dir).expanduser().resolve()
    else:
        if not args.controller:
            raise ValueError("Provide either (--logs_dir AND --runconfigs_dir) OR --controller.")
        logs_dir = (script_dir / ".." / "logs" / args.controller).resolve()
        runconfigs_dir = (script_dir / ".." / "runconfigs" / args.controller).resolve()

    if not logs_dir.exists():
        raise FileNotFoundError(f"logs_dir not found: {logs_dir}")
    if not runconfigs_dir.exists():
        raise FileNotFoundError(f"runconfigs_dir not found: {runconfigs_dir}")

    log_files = sorted(logs_dir.glob("*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No .csv logs found in: {logs_dir}")

    print(f"[data] logs_dir: {logs_dir}")
    print(f"[data] runconfigs_dir: {runconfigs_dir}")
    print(f"[data] found {len(log_files)} log files:")
    for p in log_files:
        print(f"  - {p.name}")

    episodes: List[Episode] = []
    missing_cfg: List[str] = []

    for csv_path in log_files:
        run_id = parse_run_id_from_log_name(csv_path.name)
        cfg_path = runconfigs_dir / f"{run_id}.json"
        if not cfg_path.exists():
            missing_cfg.append(f"{csv_path.name} -> expected {cfg_path.name}")
            cfg_path = None
        ep = load_episode(csv_path, cfg_path)
        episodes.append(ep)

    if missing_cfg:
        print("[warn] missing runconfig(s) for some logs; defaults used for those episodes:")
        for s in missing_cfg:
            print(f"  - {s}")

    for i, ep in enumerate(episodes):
        e = ep.env_np
        print(
            f"[episode {i:02d}] {ep.csv_path.name} | run_id={ep.run_id} | "
            f"tau_thrust={e['tau_thrust']:.4f} tilt_max_deg={e['tilt_max']*180/math.pi:.2f} "
            f"yaw_rate_max_dps={e['yaw_rate_max']*180/math.pi:.1f} thrust=[{e['thrust_min']:.2f},{e['thrust_max']:.2f}] "
            f"mass={e['mass']:.3f} drag={e['drag']:.3f}"
        )

    mds = build_multi_dataset(episodes)

    key = jax.random.PRNGKey(args.seed)
    params = init_mlp_params(key, in_dim=17, hidden=args.hidden, depth=args.depth, out_dim=4)

    optimizer_imitation = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adam(args.lr),
    )
    optimizer_rollout = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adam(args.rollout_lr),
    )

    train_step_imitation = make_train_step_imitation(optimizer_imitation, mds)
    train_step_rollout = make_train_step_rollout(
        optimizer_rollout, mds,
        lambda_u=float(args.lambda_u),
        lambda_du=float(args.lambda_du),
        lambda_tilt=float(args.lambda_tilt),
    )

    rng = np.random.default_rng(args.seed)

    # Phase 1: imitation
    X_all = np.array(mds.X_all_raw, dtype=np.float32)
    U_all = np.array(mds.U_all, dtype=np.float32)
    ep_idx_all = np.array(mds.ep_id_all, dtype=np.int32)

    opt_state = optimizer_imitation.init(params)

    print(f"[data] episodes={mds.E} rollout_T={mds.Tr} dt={mds.dt:.5f}  (imitation on N={X_all.shape[0]} samples)")
    for ep in range(args.epochs_imitation):
        losses = []
        for Xb, Ub, Ib in minibatches(rng, X_all, U_all, ep_idx_all, args.batch):
            params, opt_state, loss = train_step_imitation(
                params, opt_state,
                jnp.asarray(Xb), jnp.asarray(Ub), jnp.asarray(Ib)
            )
            losses.append(float(loss))
        print(f"[imitation] epoch {ep+1:03d}/{args.epochs_imitation}  loss={float(np.mean(losses)):.6f}")

    # Phase 2: rollout
    opt_state = optimizer_rollout.init(params)
    print("[rollout] training with differentiable closed-loop rollout loss (mean across episodes)")
    print(f"[rollout] lambda_u={args.lambda_u} lambda_du={args.lambda_du} lambda_tilt={args.lambda_tilt}")
    for ep in range(args.epochs_rollout):
        params, opt_state, loss = train_step_rollout(params, opt_state)
        print(f"[rollout]   epoch {ep+1:03d}/{args.epochs_rollout}  loss={float(loss):.6f}")

    # Save
    save_path = Path(args.save).resolve()
    flat: Dict[str, np.ndarray] = {}

    for i, (W, b) in enumerate(params):
        flat[f"W_{i}"] = np.array(W)
        flat[f"b_{i}"] = np.array(b)

    flat["x_mean"] = np.array(mds.x_mean)
    flat["x_std"] = np.array(mds.x_std)

    flat["du_mean"] = np.array(mds.du_mean)
    flat["du_std"] = np.array(mds.du_std)

    flat["u_mean"] = np.array(mds.u_mean)
    flat["u_std"] = np.array(mds.u_std)

    flat["num_episodes"] = np.array([mds.E], dtype=np.int32)

    for k, v in mds.env.items():
        flat[f"env_{k}"] = np.array(v)

    np.savez(save_path, **flat)
    print(f"[save] wrote: {save_path}")

    # Eval: teacher-in-plant vs surrogate rollout, per episode
    print("[eval] per-episode RMS position error:")

    env_keys = list(mds.env.keys())

    def eval_episode(i: int) -> Tuple[float, float]:
        env_ep = {k: mds.env[k][i] for k in env_keys}

        # teacher rollout in our plant
        states_teacher = rollout_teacher_in_our_plant(mds.state0s[i], mds.teacher_us[i], env_ep, mds.dt)
        pT = np.array(states_teacher[:, 0:3])
        pref = np.array(mds.ref_seqs[i, :, 0:3])
        rms_teacher = float(np.sqrt(np.mean(np.sum((pT - pref) ** 2, axis=-1))))

        # surrogate rollout in our plant
        u0 = clamp_u(mds.teacher_us[i, 0], env_ep)
        states, _, _ = rollout_one_delta(
            params, mds.state0s[i], mds.ref_seqs[i], env_ep, mds.dt,
            mds.x_mean, mds.x_std, mds.du_mean, mds.du_std,
            u0=u0
        )
        p = np.array(states[:, 0:3])
        rms_sur = float(np.sqrt(np.mean(np.sum((p - pref) ** 2, axis=-1))))

        return rms_teacher, rms_sur

    rms_teacher_list = []
    rms_sur_list = []

    for i in range(mds.E):
        rt, rs = eval_episode(i)
        rms_teacher_list.append(rt)
        rms_sur_list.append(rs)
        print(f"  - episode {i:02d} | teacher-in-plant: {rt:.4f} m | surrogate: {rs:.4f} m")

    print(f"[eval] mean teacher-in-plant RMS: {float(np.mean(rms_teacher_list)):.4f} m")
    print(f"[eval] mean surrogate RMS:        {float(np.mean(rms_sur_list)):.4f} m")


    # ------------------------------------------------------------
    # Diagnostics: rollout comparison (teacher-in-plant vs surrogate) + export CSV/PNG
    # ------------------------------------------------------------
    import os
    os.makedirs(args.diag_out, exist_ok=True)

    ep = int(args.diag_episode)
    if ep < 0 or ep >= mds.E:
        raise ValueError(f"--diag_episode out of range: {ep} (have {mds.E} episodes)")

    env_keys = list(mds.env.keys())
    env_ep = {k: mds.env[k][ep] for k in env_keys}

    T = int(mds.Tr)
    t = np.arange(T, dtype=np.float32) * float(mds.dt)

    # Reference (T,4)
    ref_seq = np.array(mds.ref_seqs[ep], dtype=np.float32)

    # Teacher commands (T,4) from log, then clamped inside rollout
    u_teacher_logged = mds.teacher_us[ep]

    # Teacher-in-our-plant rollout (states + clamped actions)
    states_T, u_T = rollout_teacher_in_our_plant_with_u(
        mds.state0s[ep],
        u_teacher_logged,
        env_ep,
        float(mds.dt),
    )

    # Surrogate rollout (delta-u controller)
    u0 = clamp_u(u_teacher_logged[0], env_ep)  # same anchor used in training/eval
    states_S, u_S, dU_S = rollout_one_delta(
        params,
        mds.state0s[ep],
        mds.ref_seqs[ep],
        env_ep,
        float(mds.dt),
        mds.x_mean, mds.x_std,
        mds.du_mean, mds.du_std,
        u0=u0,
    )

    # Convert to numpy for exporting/plotting
    states_T_np = np.array(states_T, dtype=np.float32)
    u_T_np = np.array(u_T, dtype=np.float32)

    states_S_np = np.array(states_S, dtype=np.float32)
    u_S_np = np.array(u_S, dtype=np.float32)
    dU_S_np = np.array(dU_S, dtype=np.float32)

    # Plot XYZ (teacher vs surrogate)
    p_teacher = states_T_np[:, 0:3]
    p_model   = states_S_np[:, 0:3]
    plot_xyz(
        os.path.join(args.diag_out, f"episode_{ep:02d}_xyz.png"),
        t, p_teacher, p_model
    )

    # Write combined CSV (one file, everything aligned row-by-row)
    csv_path = os.path.join(args.diag_out, f"episode_{ep:02d}_teacher_vs_surrogate.csv")
    write_diag_csv_combined(
        csv_path,
        t,
        ref_seq,
        states_T_np,
        u_T_np,
        states_S_np,
        u_S_np,
        dU_S_np,
    )

    print(f"[diag] wrote: {csv_path}")
    print(f"[diag] wrote: {os.path.join(args.diag_out, f'episode_{ep:02d}_xyz.png')}")



if __name__ == "__main__":
    main()
