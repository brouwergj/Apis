#!/usr/bin/env python
"""
Brood baseline (Approach A): Black-box surrogate controller trained with rollout gradients.

- Inputs:  state_t (p,v,rpy) + ref_t (p_ref, yaw_ref)
- Output:  u_t (roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd)
- Training:
    Phase 1: one-step imitation (MSE on commands)
    Phase 2: closed-loop rollout loss (tracking + smoothness + tilt regularizer)
- Differentiable rollout via jax.lax.scan
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

import jax
import jax.numpy as jnp
import optax


# ----------------------------
# Utilities
# ----------------------------

def wrap_pi(a: jnp.ndarray) -> jnp.ndarray:
    # Stable wrap to [-pi, pi]
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


# ----------------------------
# Data
# ----------------------------

@dataclass(frozen=True)
class Dataset:
    # teacher / logged sequences
    state0: jnp.ndarray        # (S,) initial state for rollout
    ref_seq: jnp.ndarray       # (T,4) p_ref(3), yaw_ref(1)
    teacher_u: jnp.ndarray     # (T,4) roll,pitch,yaw_rate,thrust (for imitation, optional)

    # per-step logged state (for one-step imitation)
    X_one: jnp.ndarray         # (T, in_dim) normalized
    Y_one: jnp.ndarray         # (T, 4) normalized (or raw if you want)

    # normalizers
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    # env constants (python dict of scalars/arrays converted later)
    env_np: Dict[str, Any]
    dt: float


def load_csv_log(csv_path: Path) -> Dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Required columns based on your logger header
    req = [
        "t",
        "px","py","pz",
        "vx","vy","vz",
        "roll","pitch","yaw",
        "prefx","prefy","prefz",
        "yaw_ref",
        "roll_cmd","pitch_cmd","yaw_rate_cmd","thrust_cmd",
        "mass","drag","g"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing}")

    out = {c: df[c].to_numpy(dtype=np.float32) for c in req}
    return out


def build_dataset(csv_path: Path, runconfig_path: Path | None) -> Dataset:
    log = load_csv_log(csv_path)

    # Build sequences
    T = log["t"].shape[0]
    # state: p(3), v(3), rpy(3), thrust_actual(1)  -> we include thrust_actual for dynamics match
    # Initialize thrust_actual from first logged thrust_cmd (reasonable) / or 0.0
    thrust0 = float(log["thrust_cmd"][0])
    state0 = np.array([
        log["px"][0], log["py"][0], log["pz"][0],
        log["vx"][0], log["vy"][0], log["vz"][0],
        log["roll"][0], log["pitch"][0], log["yaw"][0],
        thrust0,
    ], dtype=np.float32)

    ref_seq = np.stack([log["prefx"], log["prefy"], log["prefz"], log["yaw_ref"]], axis=1).astype(np.float32)
    teacher_u = np.stack([log["roll_cmd"], log["pitch_cmd"], log["yaw_rate_cmd"], log["thrust_cmd"]], axis=1).astype(np.float32)

    # Build one-step inputs X: use current state (p,v,rpy) + ref (p_ref,yaw_ref)
    # For one-step imitation, we use the logged state per tick (not the rolled-out state).
    state_seq = np.stack([
        log["px"], log["py"], log["pz"],
        log["vx"], log["vy"], log["vz"],
        log["roll"], log["pitch"], log["yaw"],
    ], axis=1).astype(np.float32)

    X_raw = np.concatenate([state_seq, ref_seq], axis=1)  # (T, 9+4=13)
    Y_raw = teacher_u  # (T,4)

    # Normalization
    x_mean = X_raw.mean(axis=0)
    x_std = safe_std(X_raw)
    y_mean = Y_raw.mean(axis=0)
    y_std = safe_std(Y_raw)

    X_one = (X_raw - x_mean) / x_std
    Y_one = (Y_raw - y_mean) / y_std

    # dt: infer from t column
    if T >= 2:
        dt = float(log["t"][1] - log["t"][0])
    else:
        dt = 0.01

    # Env: from log averages + optional RunConfig extras (taus, wind, limits)
    mass = float(np.mean(log["mass"]))
    drag = float(np.mean(log["drag"]))
    g = float(np.mean(log["g"]))

    # Defaults match your controller/dynamics file; can be overridden by RunConfig.json if present
    env = {
        "mass": mass,
        "drag": drag,
        "g": g,
        "tau_roll": 0.08,
        "tau_pitch": 0.08,
        "tau_thrust": 0.05,
        "wind": np.array([0.0, 0.0, 0.0], dtype=np.float32),

        # command bounds (used for squashing)
        "tilt_max": float(25.0 * math.pi / 180.0),
        "yaw_rate_max": float(180.0 * math.pi / 180.0),
        "thrust_min": 0.0,
        "thrust_max": 30.0,
    }

    if runconfig_path is not None and runconfig_path.exists():
        cfg = json.loads(runconfig_path.read_text(encoding="utf-8"))
        dyn = cfg.get("dynamics", {})
        ctrl = cfg.get("controller", {})

        env["tau_roll"] = float(dyn.get("tau_roll", env["tau_roll"]))
        env["tau_pitch"] = float(dyn.get("tau_pitch", env["tau_pitch"]))
        env["tau_thrust"] = float(dyn.get("tau_thrust", env["tau_thrust"]))

        wind = dyn.get("wind", {})
        if isinstance(wind, dict) and wind.get("type") == "constant":
            vec = wind.get("vec", [0.0, 0.0, 0.0])
            env["wind"] = np.array(vec, dtype=np.float32)

        env["tilt_max"] = float(float(ctrl.get("tilt_max_deg", 25.0)) * math.pi / 180.0)
        env["yaw_rate_max"] = float(float(ctrl.get("yaw_rate_max_dps", 180.0)) * math.pi / 180.0)
        env["thrust_min"] = float(ctrl.get("thrust_min", 0.0))
        env["thrust_max"] = float(ctrl.get("thrust_max", 30.0))

    return Dataset(
        state0=jnp.asarray(state0),
        ref_seq=jnp.asarray(ref_seq),
        teacher_u=jnp.asarray(teacher_u),
        X_one=jnp.asarray(X_one),
        Y_one=jnp.asarray(Y_one),
        x_mean=jnp.asarray(x_mean),
        x_std=jnp.asarray(x_std),
        y_mean=jnp.asarray(y_mean),
        y_std=jnp.asarray(y_std),
        env_np=env,
        dt=dt,
    )


# ----------------------------
# Black-box MLP controller (Approach A)
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


def controller_A(params: list, state: jax.Array, ref_t: jax.Array, env: Dict[str, jax.Array],
                 x_mean: jax.Array, x_std: jax.Array) -> jax.Array:
    """
    state: (10,) [p(3), v(3), rpy(3), thrust_actual]
    ref_t: (4,)  [p_ref(3), yaw_ref]
    returns u: (4,) [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
    """
    # Use only p,v,rpy (not thrust_actual) + ref
    x_raw = jnp.concatenate([state[0:9], ref_t])  # (13,)
    x = (x_raw - x_mean) / x_std

    y = mlp_apply(params, x)  # unconstrained (4,)

    tilt_max = env["tilt_max"]
    yaw_rate_max = env["yaw_rate_max"]
    thrust_min = env["thrust_min"]
    thrust_max = env["thrust_max"]

    roll = tilt_max * jnp.tanh(y[0])
    pitch = tilt_max * jnp.tanh(y[1])
    yaw_rate = yaw_rate_max * jnp.tanh(y[2])

    thrust_01 = 0.5 * (jnp.tanh(y[3]) + 1.0)
    thrust = thrust_min + thrust_01 * (thrust_max - thrust_min)

    return jnp.array([roll, pitch, yaw_rate, thrust], dtype=jnp.float32)


# ----------------------------
# Dynamics (JAX port with thrust_actual in state for smoother rollouts)
# ----------------------------

def dynamics_step(state: jax.Array, u: jax.Array, env: Dict[str, jax.Array], dt: float) -> jax.Array:
    """
    state: (10,) [p(3), v(3), roll, pitch, yaw, thrust_actual]
    u:     (4,)  [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]
    """
    p = state[0:3]
    v = state[3:6]
    roll, pitch, yaw = state[6], state[7], state[8]
    thrust_actual = state[9]

    roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd = u

    mass = env["mass"]
    drag = env["drag"]
    g = env["g"]
    tau_roll = env["tau_roll"]
    tau_pitch = env["tau_pitch"]
    tau_thrust = env["tau_thrust"]
    wind = env["wind"]

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
# Rollout and losses
# ----------------------------

def rollout(params: list, state0: jax.Array, ref_seq: jax.Array, env: Dict[str, jax.Array],
            dt: float, x_mean: jax.Array, x_std: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    returns:
      states: (T,10)
      us:     (T,4)
    """
    def step_fn(carry, ref_t):
        state = carry
        u = controller_A(params, state, ref_t, env, x_mean, x_std)
        nxt = dynamics_step(state, u, env, dt)
        return nxt, (state, u)

    _, (states, us) = jax.lax.scan(step_fn, state0, ref_seq)
    return states, us


def one_step_imitation_loss(params: list, Xb: jax.Array, Yb: jax.Array) -> jax.Array:
    pred = jax.vmap(lambda x: mlp_apply(params, x))(Xb)
    return jnp.mean((pred - Yb) ** 2)


def rollout_loss(params: list, ds: Dataset, env: Dict[str, jax.Array]) -> jax.Array:
    states, us = rollout(params, ds.state0, ds.ref_seq, env, ds.dt, ds.x_mean, ds.x_std)

    # Tracking (position) loss
    p = states[:, 0:3]
    p_ref = ds.ref_seq[:, 0:3]
    track = jnp.mean(jnp.sum((p - p_ref) ** 2, axis=-1))

    # Smoothness of control
    du = us[1:] - us[:-1]
    smooth = jnp.mean(jnp.sum(du ** 2, axis=-1))

    # Tilt regularizer
    tilt = jnp.mean(jnp.sum(states[:, 6:8] ** 2, axis=-1))

    return track + 1e-3 * smooth + 1e-3 * tilt


def make_train_step_imitation(optimizer):
    @jax.jit
    def step(params, opt_state, Xb, Yb):
        loss, grads = jax.value_and_grad(one_step_imitation_loss)(params, Xb, Yb)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def make_train_step_rollout(optimizer, ds: Dataset, env: Dict[str, jax.Array]):
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(rollout_loss)(params, ds, env)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step



# ----------------------------
# Main
# ----------------------------

def make_env_jax(env_np: Dict[str, Any]) -> Dict[str, jax.Array]:
    env = {}
    for k, v in env_np.items():
        if isinstance(v, np.ndarray):
            env[k] = jnp.asarray(v, dtype=jnp.float32)
        else:
            env[k] = jnp.asarray(v, dtype=jnp.float32)
    return env


def minibatches(rng: np.random.Generator, X: np.ndarray, Y: np.ndarray, batch_size: int):
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for i in range(0, N, batch_size):
        j = idx[i:i + batch_size]
        yield X[j], Y[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", type=str, required=True,
                    help="Controller type folder name under ../logs and ../runconfigs")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs_imitation", type=int, default=10)
    ap.add_argument("--epochs_rollout", type=int, default=30)

    ap.add_argument("--save", type=str, default="brood_baseline_A_params.npz")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    logs_dir = (script_dir / ".." / "logs" / args.controller).resolve()
    runconfigs_dir = (script_dir / ".." / "runconfigs" / args.controller).resolve()

    if not logs_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {logs_dir}")

    log_files = sorted(logs_dir.glob("*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No .csv logs found in: {logs_dir}")

    csv_path = log_files[0]
    rc_path = runconfigs_dir / "RunConfig.json"
    print(rc_path)
    print(f"[data] log: {csv_path}")
    print(f"[data] runconfig: {rc_path if rc_path.exists() else '(missing)'}")

    ds = build_dataset(csv_path, rc_path if (rc_path and rc_path.exists()) else None)
    env = make_env_jax(ds.env_np)

    in_dim = int(ds.X_one.shape[1])  # 13 normalized
    out_dim = 4

    key = jax.random.PRNGKey(args.seed)
    params = init_mlp_params(key, in_dim=in_dim, hidden=args.hidden, depth=args.depth, out_dim=out_dim)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    train_step_imitation = make_train_step_imitation(optimizer)
    train_step_rollout = make_train_step_rollout(optimizer, ds, env)

    # Phase 1: one-step imitation warmup
    rng = np.random.default_rng(args.seed)
    X_np = np.array(ds.X_one, dtype=np.float32)
    Y_np = np.array(ds.Y_one, dtype=np.float32)

    print(f"[data] T={X_np.shape[0]} dt={ds.dt:.5f}  (warmup: one-step imitation)")
    for ep in range(args.epochs_imitation):
        losses = []
        for Xb, Yb in minibatches(rng, X_np, Y_np, args.batch):
            params, opt_state, loss = train_step_imitation(
                params, opt_state, jnp.asarray(Xb), jnp.asarray(Yb)
            )
            losses.append(float(loss))
        print(f"[imitation] epoch {ep+1:03d}/{args.epochs_imitation}  loss={np.mean(losses):.6f}")

    # Phase 2: rollout loss (the “grad for free” step)
    print("[rollout] training with differentiable closed-loop rollout loss")
    for ep in range(args.epochs_rollout):
        params, opt_state, loss = train_step_rollout(params, opt_state)
        print(f"[rollout]   epoch {ep+1:03d}/{args.epochs_rollout}  loss={float(loss):.6f}")

    # Save params + normalizers so you can run inference later
    save_path = Path(args.save)
    flat = {}
    for i, (W, b) in enumerate(params):
        flat[f"W_{i}"] = np.array(W)
        flat[f"b_{i}"] = np.array(b)

    flat["x_mean"] = np.array(ds.x_mean)
    flat["x_std"] = np.array(ds.x_std)
    flat["y_mean"] = np.array(ds.y_mean)
    flat["y_std"] = np.array(ds.y_std)

    # Save env scalars
    for k, v in ds.env_np.items():
        flat[f"env_{k}"] = np.array(v, dtype=np.float32)

    np.savez(save_path, **flat)
    print(f"[save] wrote: {save_path}")

    # Quick sanity: rollout once and report final tracking error
    states, us = rollout(params, ds.state0, ds.ref_seq, env, ds.dt, ds.x_mean, ds.x_std)
    p = np.array(states[:, 0:3])
    pref = np.array(ds.ref_seq[:, 0:3])
    rms = np.sqrt(np.mean(np.sum((p - pref) ** 2, axis=-1)))
    print(f"[eval] rollout RMS position error: {rms:.4f} m")


if __name__ == "__main__":
    main()
