#!/usr/bin/env python
"""
Brood baseline (Approach A): Black-box surrogate controller trained with rollout gradients.

Multi-episode training + per-episode RunConfig matching.

For each log file:
  logs_dir/ep_<name>_YYYYMMDD_HHMMSS.csv
we look for:
  runconfigs_dir/ep_<name>.json

We read only what matters from RunConfig for the estimator:
  dynamics: tau_roll, tau_pitch, tau_thrust, wind
  controller: tilt_max_deg, yaw_rate_max_dps, thrust_min, thrust_max

From the CSV we read the rest:
  state, references, teacher commands, mass/drag/g (episode-specific).

Training:
  Phase 1: imitation on physical bounded controller outputs vs teacher_u (uses per-sample episode env).
  Phase 2: differentiable closed-loop rollout loss averaged across episodes (uses per-episode env).

Notes:
- Rollout arrays are cropped to the minimum length across episodes (Tr) to keep shapes static for JAX.
- Imitation uses all samples across all episodes (no cropping).
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

    If name doesn't match pattern, fall back to stem before first timestamp-like suffix.
    """
    stem = Path(csv_name).stem  # without .csv
    parts = stem.split("_")
    if len(parts) >= 3:
        maybe_date = parts[-2]
        maybe_time = parts[-1]
        if maybe_date.isdigit() and len(maybe_date) == 8 and maybe_time.isdigit() and len(maybe_time) == 6:
            return "_".join(parts[:-2])
    # fallback: whole stem
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

    # rollout sequences
    state0: np.ndarray      # (10,)
    ref_seq: np.ndarray     # (T,4)
    teacher_u: np.ndarray   # (T,4)

    # imitation raw inputs
    X_raw: np.ndarray       # (T,13) = state(9)+ref(4)

    # env scalars (episode-specific)
    env_np: Dict[str, Any]


@dataclass(frozen=True)
class MultiDataset:
    # Episode-major arrays (cropped to common rollout length Tr)
    state0s: jnp.ndarray        # (E,10)
    ref_seqs: jnp.ndarray       # (E,Tr,4)
    teacher_us: jnp.ndarray     # (E,Tr,4)

    # Global normalizers (computed across ALL imitation samples)
    x_mean: jnp.ndarray         # (13,)
    x_std: jnp.ndarray          # (13,)
    y_mean: jnp.ndarray         # (4,)
    y_std: jnp.ndarray          # (4,)

    # Imitation training arrays (all samples)
    X_all_raw: jnp.ndarray      # (N,13)
    U_all: jnp.ndarray          # (N,4)
    ep_id_all: jnp.ndarray      # (N,) int32 episode index per sample

    # Env arrays per episode
    env: Dict[str, jnp.ndarray] # values are (E,) or (E,3)

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
# RunConfig reading (only what estimator needs)
# ----------------------------

def read_env_from_runconfig(runconfig_path: Optional[Path]) -> Dict[str, Any]:
    """
    Returns estimator-relevant env fields with defaults.
    If runconfig is missing, defaults are used.
    """
    env = {
        # actuator/inner-loop lags
        "tau_roll": 0.08,
        "tau_pitch": 0.08,
        "tau_thrust": 0.05,

        # disturbance model (optional)
        "wind": np.array([0.0, 0.0, 0.0], dtype=np.float32),

        # output limits
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

    # state0: p(3), v(3), rpy(3), thrust_actual(1)
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

    # env from runconfig (taus/limits/wind) + from CSV (mass/drag/g)
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

    # GLOBAL normalizers across all imitation samples
    X_all = np.concatenate([ep.X_raw for ep in episodes], axis=0).astype(np.float32)         # (N,13)
    U_all = np.concatenate([ep.teacher_u for ep in episodes], axis=0).astype(np.float32)     # (N,4)

    x_mean = X_all.mean(axis=0)
    x_std = safe_std(X_all)
    y_mean = U_all.mean(axis=0)
    y_std = safe_std(U_all)

    # Episode ids per sample
    ep_ids = []
    for i, ep in enumerate(episodes):
        ep_ids.append(np.full((ep.X_raw.shape[0],), i, dtype=np.int32))
    ep_id_all = np.concatenate(ep_ids, axis=0)

    # Episode-major (cropped) arrays
    state0s = np.stack([ep.state0 for ep in episodes], axis=0).astype(np.float32)               # (E,10)
    ref_seqs = np.stack([ep.ref_seq[:Tr] for ep in episodes], axis=0).astype(np.float32)        # (E,Tr,4)
    teacher_us = np.stack([ep.teacher_u[:Tr] for ep in episodes], axis=0).astype(np.float32)    # (E,Tr,4)

    # Env per episode -> dict of arrays
    env_keys = episodes[0].env_np.keys()
    env: Dict[str, np.ndarray] = {}
    for k in env_keys:
        vals = [ep.env_np[k] for ep in episodes]
        if isinstance(vals[0], np.ndarray):
            env[k] = np.stack(vals, axis=0).astype(np.float32)   # (E,3)
        else:
            env[k] = np.array(vals, dtype=np.float32)            # (E,)

    return MultiDataset(
        state0s=jnp.asarray(state0s),
        ref_seqs=jnp.asarray(ref_seqs),
        teacher_us=jnp.asarray(teacher_us),
        x_mean=jnp.asarray(x_mean),
        x_std=jnp.asarray(x_std),
        y_mean=jnp.asarray(y_mean),
        y_std=jnp.asarray(y_std),
        X_all_raw=jnp.asarray(X_all),
        U_all=jnp.asarray(U_all),
        ep_id_all=jnp.asarray(ep_id_all),
        env={k: jnp.asarray(v) for k, v in env.items()},
        dt=dt0,
        Tr=Tr,
        E=E,
    )


# ----------------------------
# MLP controller
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


def controller_A(
    params: list,
    state: jax.Array,               # (10,)
    ref_t: jax.Array,               # (4,)
    env_ep: Dict[str, jax.Array],   # scalars for one episode
    x_mean: jax.Array,
    x_std: jax.Array,
    y_mean: jax.Array,
    y_std: jax.Array,
) -> jax.Array:
    """
    Returns physical bounded commands u = [roll, pitch, yaw_rate, thrust]
    """
    x_raw = jnp.concatenate([state[0:9], ref_t])  # (13,)
    x = (x_raw - x_mean) / x_std

    y_norm = mlp_apply(params, x)
    y = y_norm * y_std + y_mean  # physical units

    tilt_max = env_ep["tilt_max"]
    yaw_rate_max = env_ep["yaw_rate_max"]
    thrust_min = env_ep["thrust_min"]
    thrust_max = env_ep["thrust_max"]

    roll = jnp.clip(y[0], -tilt_max, tilt_max)
    pitch = jnp.clip(y[1], -tilt_max, tilt_max)
    yaw_rate = jnp.clip(y[2], -yaw_rate_max, yaw_rate_max)
    thrust = jnp.clip(y[3], thrust_min, thrust_max)

    return jnp.array([roll, pitch, yaw_rate, thrust], dtype=jnp.float32)


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
# Rollout + losses
# ----------------------------

def rollout_one(
    params: list,
    state0: jax.Array,              # (10,)
    ref_seq: jax.Array,             # (Tr,4)
    env_ep: Dict[str, jax.Array],
    dt: float,
    x_mean: jax.Array,
    x_std: jax.Array,
    y_mean: jax.Array,
    y_std: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    def step_fn(carry, ref_t):
        state = carry
        u = controller_A(params, state, ref_t, env_ep, x_mean, x_std, y_mean, y_std)
        nxt = dynamics_step(state, u, env_ep, dt)
        return nxt, (state, u)

    _, (states, us) = jax.lax.scan(step_fn, state0, ref_seq)
    return states, us


def one_step_imitation_loss(
    params: list,
    Xb_raw: jax.Array,              # (B,13)
    Ub_teacher: jax.Array,          # (B,4)
    ep_idx: jax.Array,              # (B,) int32
    mds: MultiDataset,
) -> jax.Array:
    """
    Imitation compares physical bounded controller outputs to teacher commands,
    using per-sample episode env.
    """
    env_keys = list(mds.env.keys())

    def pred_u(x_raw, u_teacher, i):
        i = i.astype(jnp.int32)
        env_ep = {k: mds.env[k][i] for k in env_keys}

        state9 = x_raw[0:9]
        ref_t = x_raw[9:13]

        # thrust_actual proxy for one-step: use teacher thrust at that step
        thrust_actual = u_teacher[3]
        state = jnp.concatenate([state9, jnp.array([thrust_actual], dtype=jnp.float32)])

        return controller_A(params, state, ref_t, env_ep, mds.x_mean, mds.x_std, mds.y_mean, mds.y_std)

    u_hat = jax.vmap(pred_u)(Xb_raw, Ub_teacher, ep_idx)
    return jnp.mean((u_hat - Ub_teacher) ** 2)


def rollout_loss(params: list, mds: MultiDataset) -> jax.Array:
    """
    Average rollout loss across episodes (uses each episode's env).
    """
    env_keys = list(mds.env.keys())

    def loss_one(i: jax.Array) -> jax.Array:
        i = i.astype(jnp.int32)
        env_ep = {k: mds.env[k][i] for k in env_keys}

        states, us = rollout_one(
            params,
            mds.state0s[i],
            mds.ref_seqs[i],
            env_ep,
            mds.dt,
            mds.x_mean, mds.x_std,
            mds.y_mean, mds.y_std,
        )

        p = states[:, 0:3]
        p_ref = mds.ref_seqs[i, :, 0:3]
        track = jnp.mean(jnp.sum((p - p_ref) ** 2, axis=-1))

        du = us[1:] - us[:-1]
        smooth = jnp.mean(jnp.sum(du ** 2, axis=-1))

        tilt = jnp.mean(jnp.sum(states[:, 6:8] ** 2, axis=-1))

        return track + 1e-3 * smooth + 1e-3 * tilt

    losses = jax.vmap(loss_one)(jnp.arange(mds.E))
    return jnp.mean(losses)


def make_train_step_imitation(optimizer, mds: MultiDataset):
    @jax.jit
    def step(params, opt_state, Xb_raw, Ub_teacher, ep_idx):
        loss, grads = jax.value_and_grad(one_step_imitation_loss)(params, Xb_raw, Ub_teacher, ep_idx, mds)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def make_train_step_rollout(optimizer, mds: MultiDataset):
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(rollout_loss)(params, mds)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


# ----------------------------
# Main
# ----------------------------

def minibatches(rng: np.random.Generator, X: np.ndarray, U: np.ndarray, ep_idx: np.ndarray, batch_size: int):
    """
    Fixed-size batches only (skip last partial) to avoid JIT recompiles.
    """
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for i in range(0, N, batch_size):
        j = idx[i:i + batch_size]
        if j.shape[0] != batch_size:
            continue
        yield X[j], U[j], ep_idx[j]


def main():
    ap = argparse.ArgumentParser()

    # Prefer explicit dirs
    ap.add_argument("--logs_dir", type=str, default=None,
                    help="Directory containing *.csv logs (preferred)")
    ap.add_argument("--runconfigs_dir", type=str, default=None,
                    help="Directory containing matching *.json runconfigs (preferred)")

    # Legacy convenience: controller name implies ../logs/<controller> and ../runconfigs/<controller>
    ap.add_argument("--controller", type=str, default=None,
                    help="Controller folder name under ../logs and ../runconfigs (legacy)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--lr", type=float, default=1e-3, help="LR for imitation phase")
    ap.add_argument("--rollout_lr", type=float, default=1e-4, help="LR for rollout phase")
    ap.add_argument("--clip_norm", type=float, default=1.0)

    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs_imitation", type=int, default=10)
    ap.add_argument("--epochs_rollout", type=int, default=30)

    ap.add_argument("--save", type=str, default="brood_baseline_A_params.npz")
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

    # Build episodes with per-log runconfig match
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

    # Print per-episode env summary (useful sanity)
    for i, ep in enumerate(episodes):
        e = ep.env_np
        print(
            f"[episode {i:02d}] {ep.csv_path.name} | run_id={ep.run_id} | "
            f"tau_thrust={e['tau_thrust']:.4f} tilt_max_deg={e['tilt_max']*180/math.pi:.2f} "
            f"yaw_rate_max_dps={e['yaw_rate_max']*180/math.pi:.1f} thrust=[{e['thrust_min']:.2f},{e['thrust_max']:.2f}] "
            f"mass={e['mass']:.3f} drag={e['drag']:.3f}"
        )

    mds = build_multi_dataset(episodes)

    # Init model
    key = jax.random.PRNGKey(args.seed)
    params = init_mlp_params(key, in_dim=13, hidden=args.hidden, depth=args.depth, out_dim=4)

    optimizer_imitation = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adam(args.lr),
    )
    optimizer_rollout = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adam(args.rollout_lr),
    )

    train_step_imitation = make_train_step_imitation(optimizer_imitation, mds)
    train_step_rollout = make_train_step_rollout(optimizer_rollout, mds)

    # --------------------
    # Phase 1: imitation on all samples across episodes
    # --------------------
    rng = np.random.default_rng(args.seed)
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

    # --------------------
    # Phase 2: rollout loss averaged across episodes
    # --------------------
    opt_state = optimizer_rollout.init(params)  # reset optimizer state

    print("[rollout] training with differentiable closed-loop rollout loss (mean across episodes)")
    for ep in range(args.epochs_rollout):
        params, opt_state, loss = train_step_rollout(params, opt_state)
        print(f"[rollout]   epoch {ep+1:03d}/{args.epochs_rollout}  loss={float(loss):.6f}")

    # --------------------
    # Save params + global normalizers + per-episode env (for reproducibility)
    # --------------------
    save_path = Path(args.save).resolve()
    flat: Dict[str, np.ndarray] = {}

    for i, (W, b) in enumerate(params):
        flat[f"W_{i}"] = np.array(W)
        flat[f"b_{i}"] = np.array(b)

    flat["x_mean"] = np.array(mds.x_mean)
    flat["x_std"] = np.array(mds.x_std)
    flat["y_mean"] = np.array(mds.y_mean)
    flat["y_std"] = np.array(mds.y_std)

    flat["num_episodes"] = np.array([mds.E], dtype=np.int32)

    # store per-episode env arrays
    for k, v in mds.env.items():
        flat[f"env_{k}"] = np.array(v)

    np.savez(save_path, **flat)
    print(f"[save] wrote: {save_path}")

    # --------------------
    # Eval: per-episode RMS position error
    # --------------------
    env_keys = list(mds.env.keys())

    def eval_one(i: int) -> float:
        env_ep = {k: mds.env[k][i] for k in env_keys}
        states, us = rollout_one(
            params, mds.state0s[i], mds.ref_seqs[i], env_ep, mds.dt,
            mds.x_mean, mds.x_std, mds.y_mean, mds.y_std
        )
        p = np.array(states[:, 0:3])
        pref = np.array(mds.ref_seqs[i, :, 0:3])
        return float(np.sqrt(np.mean(np.sum((p - pref) ** 2, axis=-1))))

    rms_list = [eval_one(i) for i in range(mds.E)]
    for i, r in enumerate(rms_list):
        print(f"[eval] episode {i:02d} RMS pos err: {r:.4f} m")
    print(f"[eval] mean RMS pos err: {float(np.mean(rms_list)):.4f} m")


if __name__ == "__main__":
    main()
