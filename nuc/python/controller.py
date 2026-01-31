# controller.py
# - WebSocket control: receives one START JSON from Unity
# - Runs one episode of a 3D toy quad + selected controller (toy or Crazyflie PID)
# - Logs every sim tick to CSV
# - Optionally streams UDP telemetry at a separate rate
# - Sends DONE with log path and exits

import asyncio
import csv
import json
import math
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import websockets

from toy_controller.toy_controller import (
    CascadedController,
    QuadState,
    ReferenceProgram,
    ToyQuadDynamics,
    clamp,
    rpy_to_rotmat,
    rotmat_to_quat,
    sample_init,
    ensure_dir,
)
from crazyflie_controller.crazyflie_controller import CrazyflieController
from simple_controller.simple_controller import SimpleController


HOST = "127.0.0.1"
PORT = 7361


# controller.py (replace _select_controller with this)
def _select_controller(cfg: Dict[str, Any], dt: float):
    ctrl_cfg = cfg.get("controller", {}) if isinstance(cfg, dict) else {}
    ctrl_type = str(ctrl_cfg.get("type", cfg.get("controller_type", "toy_controller"))).lower()

    if ctrl_type in ("crazyflie_controller"):
        return "crazyflie_controller", CrazyflieController(ctrl_cfg, dt)

    if ctrl_type in ("simple_controller"):
        return "simple_controller", SimpleController(ctrl_cfg)

    if ctrl_type in ("toy_controller"):
        return "toy_controller", CascadedController(ctrl_cfg)



def run_episode(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    run_id = cfg["run_id"]
    seed = int(cfg["seed"])
    dt = float(cfg["dt"])
    duration_s = float(cfg["duration_s"])
    steps = int(math.ceil(duration_s / dt))

    real_time = bool(cfg.get("real_time", False))

    rng = np.random.default_rng(seed)

    # Dynamics randomization (light)
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
        wind_vec = wind.get("vec", [0.0, 0.0, 0.0])

    ctrl_name, controller = _select_controller(cfg, dt)

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
    flight_plan = cfg.get("flight_plan", [])
    refprog = ReferenceProgram(flight_plan, rng)
    refprog.reset()

    # Init
    state = sample_init(cfg.get("init", {}), rng)

    # Logging
    log_cfg = cfg.get("logging", {})
    out_dir = log_cfg.get("out_dir", "logs")
    ensure_dir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"{run_id}_{ctrl_name}_{ts}.csv")

    # Telemetry (UDP)
    tele_cfg = cfg.get("telemetry", {"enabled": False})
    udp_enabled = bool(tele_cfg.get("enabled", False))
    udp_rate_hz = float(tele_cfg.get("rate_hz", 60.0))
    udp_host = str(tele_cfg.get("udp_host", "127.0.0.1"))
    udp_port = int(tele_cfg.get("udp_port", 15000))
    udp_payload = str(tele_cfg.get("payload", "pose"))

    sock = None
    udp_period = 1.0 / udp_rate_hz if udp_rate_hz > 0 else 0.0
    next_udp_t = 0.0
    if udp_enabled:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Run
    t = 0.0
    max_xy = 10.0
    min_z = 0.0
    max_z = 10.0

    termination_reason = "time_limit"

    wall_start = time.perf_counter()

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
            thrust_norm = clamp(float(u.thrust_cmd) / max(hover_T * 2.0, 1e-6), 0.0, 1.0)
            state = dynamics.step(state, u, mass=mass, dt=dt)

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

            # Telemetry at independent rate (best-effort)
            if udp_enabled and sock is not None and t >= next_udp_t:
                R = rpy_to_rotmat(state.roll, state.pitch, state.yaw)
                qx, qy, qz, qw = rotmat_to_quat(R)
                if udp_payload == "pose_vel":
                    payload = {
                        "t": t,
                        "p": [float(state.p[0]), float(state.p[1]), float(state.p[2])],
                        "q": [qx, qy, qz, qw],
                        "v": [float(state.v[0]), float(state.v[1]), float(state.v[2])],
                        "thrust": float(u.thrust_cmd),
                        "thrust_norm": float(thrust_norm),
                    }
                else:
                    payload = {
                        "t": t,
                        "p": [float(state.p[0]), float(state.p[1]), float(state.p[2])],
                        "q": [qx, qy, qz, qw],
                        "thrust": float(u.thrust_cmd),
                        "thrust_norm": float(thrust_norm),
                    }
                sock.sendto(json.dumps(payload).encode("utf-8"), (udp_host, udp_port))
                next_udp_t += udp_period

            # Termination conditions (simple safety box)
            if abs(state.p[0]) > max_xy or abs(state.p[1]) > max_xy:
                termination_reason = "out_of_bounds_xy"
                break
            if state.p[2] < min_z:
                termination_reason = "hit_ground"
                break
            if state.p[2] > max_z:
                termination_reason = "out_of_bounds_z"
                break
            if not np.isfinite(state.p).all() or not np.isfinite(state.v).all():
                termination_reason = "nan_guard"
                break

            t += dt

            if real_time:
                elapsed = time.perf_counter() - wall_start
                target = (k + 1) * dt
                sleep_s = target - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    if sock is not None:
        sock.close()

    summary = {
        "run_id": run_id,
        "log_path": log_path,
        "seed": seed,
        "dt": dt,
        "duration_s": duration_s,
        "steps_executed": k + 1,
        "termination_reason": termination_reason,
        "mass": mass,
        "drag": drag,
        "gravity": gravity,
        "real_time": real_time,
        "controller": ctrl_name,
    }
    return log_path, summary


async def handler(ws):
    msg = await ws.recv()
    print("Received JSON from Unity:")
    print(msg)

    cfg = json.loads(msg)

    if cfg.get("type") != "START":
        await ws.send(json.dumps({"ok": False, "error": "Expected type=START"}))
        return

    for key in ["run_id", "seed", "dt", "duration_s"]:
        if key not in cfg:
            await ws.send(json.dumps({"ok": False, "error": f"Missing required field: {key}"}))
            return

    await ws.send(json.dumps({"ok": True, "message": "ACK: starting episode"}))

    log_path, summary = run_episode(cfg)

    await ws.send(json.dumps({"ok": True, "type": "DONE", "summary": summary}))
    print(f"DONE. Log saved at: {log_path}")
    print("Summary:", json.dumps(summary, indent=2))

    asyncio.get_running_loop().call_soon(asyncio.get_running_loop().stop)


async def main():
    print(f"Controller server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        pass
    print("Server exiting.")
