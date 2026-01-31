# runner_sa.py
#
# Usage:
#   python runner_sa.py --controller-type toy_controller --runconfig "..\runconfigs\toy_controller\ep_simple_nominal_mixed.json"

#
# Runs one episode as fast as possible.
# Writes CSV logs and saves a PNG plot (x/y/z vs time with reference overlay)
# in a sibling folder called "plots" next to "logs".

import argparse
import csv
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

# Force non-interactive plotting (no windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_controller.toy_controller import (
    CascadedController,
    ReferenceProgram,
    ToyQuadDynamics,
    clamp,
    sample_init,
    ensure_dir,
)
from crazyflie_controller.crazyflie_controller import CrazyflieController
from simple_controller.simple_controller import SimpleController


# ---------- controller selection (fail-fast) ----------

def select_controller(cfg: Dict[str, Any], dt: float):
    if "controller" not in cfg or not isinstance(cfg["controller"], dict):
        raise ValueError("runconfig must contain a 'controller' object")

    ctrl_cfg = cfg["controller"]
    if "type" not in ctrl_cfg:
        raise ValueError("controller.type is required")

    ctrl_type = str(ctrl_cfg["type"]).lower()

    if ctrl_type == "crazyflie_controller":
        return "crazyflie_controller", CrazyflieController(ctrl_cfg, dt)

    if ctrl_type == "simple_controller":
        return "simple_controller", SimpleController(ctrl_cfg)

    if ctrl_type == "toy_controller":
        return "toy_controller", CascadedController(ctrl_cfg)

    raise ValueError(
        f"Unknown controller.type '{ctrl_type}'. "
        f"Valid types: crazyflie_controller, simple_controller, toy_controller"
    )


# ---------- plotting helpers ----------

def _plots_dir_next_to_logs(out_dir: str) -> str:
    """
    Create a directory named 'plots' next to the directory named 'logs'.

    Examples:
      out_dir = ".../logs" -> plots at ".../plots"
      out_dir = ".../logs/simple_controller" -> plots at ".../plots"
      out_dir = "logs" -> plots at "./plots"
      out_dir = "logs/simple_controller" -> plots at "./plots"
    """
    abs_out = os.path.abspath(out_dir)

    if os.path.basename(abs_out) == "logs":
        logs_base = abs_out
    elif os.path.basename(os.path.dirname(abs_out)) == "logs":
        logs_base = os.path.dirname(abs_out)
    else:
        # Fallback: treat out_dir as the "logs-like" directory
        logs_base = abs_out

    plots_dir = os.path.join(os.path.dirname(logs_base), "plots")
    ensure_dir(plots_dir)
    return plots_dir


def save_position_plots(
    *,
    t: np.ndarray,
    p: np.ndarray,      # shape (T, 3)
    pref: np.ndarray,   # shape (T, 3)
    plots_dir: str,
    base_name: str,
) -> str:
    """
    Saves a wide figure with 3 vertically-stacked subplots:
      x(t) with x_ref(t), y(t) with y_ref(t), z(t) with z_ref(t)
    """
    fig = plt.figure(figsize=(14, 7.5))  # wide and not too tall
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["x", "y", "z"]

    for i, ax in enumerate(axes):
        ax.plot(t, p[:, i], label=f"{labels[i]} actual")
        ax.plot(t, pref[:, i], label=f"{labels[i]} ref")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.25)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
        ax.legend(loc="upper right")

    fig.suptitle("Position tracking: actual vs reference", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(plots_dir, f"{base_name}.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


# ---------- episode runner ----------

def run_episode(cfg: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    # Required fields
    for key in ["run_id", "seed", "dt", "duration_s"]:
        if key not in cfg:
            raise ValueError(f"Missing required field: {key}")

    run_id = cfg["run_id"]
    seed = int(cfg["seed"])
    dt = float(cfg["dt"])
    duration_s = float(cfg["duration_s"])
    steps = int(math.ceil(duration_s / dt))

    rng = np.random.default_rng(seed)

    # Dynamics configuration
    dyn_cfg = cfg.get("dynamics", {})
    mass = float(dyn_cfg.get("mass", 1.0))
    if "mass_range" in dyn_cfg:
        lo, hi = dyn_cfg["mass_range"]
        mass = float(rng.uniform(lo, hi))

    drag = float(dyn_cfg.get("drag_coeff", 0.08))
    if "drag_range" in dyn_cfg:
        lo, hi = dyn_cfg["drag_range"]
        drag = float(rng.uniform(lo, hi))

    gravity = float(dyn_cfg.get("gravity", 9.81))

    wind = dyn_cfg.get("wind", {"type": "none"})
    wind_vec = [0.0, 0.0, 0.0]
    if isinstance(wind, dict) and wind.get("type") == "constant":
        wind_vec = wind.get("vec", wind_vec)

    ctrl_name, controller = select_controller(cfg, dt)

    dynamics = ToyQuadDynamics({
        "gravity": gravity,
        "drag_coeff": drag,
        "tau_roll": dyn_cfg.get("tau_roll", 0.08),
        "tau_pitch": dyn_cfg.get("tau_pitch", 0.08),
        "tau_thrust": dyn_cfg.get("tau_thrust", 0.05),
        "wind": wind_vec,
    })
    dynamics.reset()

    # Reference program
    refprog = ReferenceProgram(cfg.get("flight_plan", []), rng)
    refprog.reset()

    # Initial state
    state = sample_init(cfg.get("init", {}), rng)

    # Logging
    log_cfg = cfg.get("logging", {})
    out_dir = log_cfg.get("out_dir", "logs")
    ensure_dir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"{run_id}_{ctrl_name}_{ts}.csv")

    # Accumulate for plotting (positions + references)
    t_hist = np.zeros((steps,), dtype=np.float64)
    p_hist = np.zeros((steps, 3), dtype=np.float64)
    pref_hist = np.zeros((steps, 3), dtype=np.float64)

    termination_reason = "time_limit"
    t = 0.0

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "px","py","pz",
            "vx","vy","vz",
            "roll","pitch","yaw",
            "prefx","prefy","prefz",
            "yaw_ref",
            "roll_cmd","pitch_cmd","yaw_rate_cmd","thrust_cmd","thrust_norm",
            "mass","drag","g"
        ])

        for k in range(steps):
            ref = refprog.step(t, dt, state)

            if isinstance(controller, CrazyflieController):
                u = controller.compute(state, ref, mass=mass, g=gravity, dt=dt)
            else:
                u = controller.compute(state, ref, mass=mass, g=gravity)

            hover_T = mass * gravity
            thrust_norm = clamp(
                float(u.thrust_cmd) / max(hover_T * 2.0, 1e-6),
                0.0, 1.0
            )

            state = dynamics.step(state, u, mass=mass, dt=dt)

            # Log row
            w.writerow([
                t,
                state.p[0], state.p[1], state.p[2],
                state.v[0], state.v[1], state.v[2],
                state.roll, state.pitch, state.yaw,
                ref.p_ref[0], ref.p_ref[1], ref.p_ref[2],
                ref.yaw_ref,
                u.roll_cmd, u.pitch_cmd, u.yaw_rate_cmd, u.thrust_cmd, thrust_norm,
                mass, drag, gravity
            ])

            # Accumulate for plots
            t_hist[k] = t
            p_hist[k, :] = state.p
            pref_hist[k, :] = ref.p_ref

            # Safety termination
            if abs(state.p[0]) > 10.0 or abs(state.p[1]) > 10.0:
                termination_reason = "out_of_bounds_xy"
                steps_executed = k + 1
                break
            if state.p[2] < 0.0:
                termination_reason = "hit_ground"
                steps_executed = k + 1
                break
            if state.p[2] > 10.0:
                termination_reason = "out_of_bounds_z"
                steps_executed = k + 1
                break
            if not np.isfinite(state.p).all() or not np.isfinite(state.v).all():
                termination_reason = "nan_guard"
                steps_executed = k + 1
                break

            t += dt
        else:
            steps_executed = steps

    # Save plots next to logs
    plots_dir = _plots_dir_next_to_logs(out_dir)
    base_name = os.path.splitext(os.path.basename(log_path))[0]

    plot_path = save_position_plots(
        t=t_hist[:steps_executed],
        p=p_hist[:steps_executed],
        pref=pref_hist[:steps_executed],
        plots_dir=plots_dir,
        base_name=base_name,
    )

    summary = {
        "run_id": run_id,
        "log_path": log_path,
        "plot_path": plot_path,
        "seed": seed,
        "dt": dt,
        "duration_s": duration_s,
        "steps_executed": steps_executed,
        "termination_reason": termination_reason,
        "mass": mass,
        "drag": drag,
        "gravity": gravity,
        "controller": ctrl_name,
    }

    return log_path, plot_path, summary


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Standalone quad controller runner (CSV + plots).")
    ap.add_argument("--controller-type", required=True,
                    choices=["toy_controller", "simple_controller", "crazyflie_controller"])
    ap.add_argument("--runconfig", required=True, help="Path to runconfig JSON")
    ap.add_argument("--out-dir", default=None, help="Override logging.out_dir")

    args = ap.parse_args()

    with open(args.runconfig, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Runconfig JSON must be an object")

    # Normalize / override
    cfg["type"] = "START"
    cfg.setdefault("controller", {})
    if not isinstance(cfg["controller"], dict):
        raise ValueError("runconfig controller field must be a JSON object")
    cfg["controller"]["type"] = args.controller_type

    if args.out_dir is not None:
        cfg.setdefault("logging", {})
        if not isinstance(cfg["logging"], dict):
            raise ValueError("runconfig logging field must be a JSON object")
        cfg["logging"]["out_dir"] = args.out_dir

    log_path, plot_path, summary = run_episode(cfg)

    print(f"DONE. CSV saved at:  {log_path}")
    print(f"DONE. Plot saved at: {plot_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
