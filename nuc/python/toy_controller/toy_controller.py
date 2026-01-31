# toyquad_server_once.py
# - WebSocket control: receives one START JSON from Unity
# - Runs one episode of a 3D toy quad + cascaded controller
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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import websockets


HOST = "127.0.0.1"
PORT = 7361


# ---------------------------
# Math helpers
# ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def wrap_pi(a: float) -> float:
    # wrap to [-pi, pi]
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx

def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    # Returns (x,y,z,w). Standard conversion.
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return (qx, qy, qz, qw)


# ---------------------------
# State and config
# ---------------------------

@dataclass
class QuadState:
    p: np.ndarray  # position (3,)
    v: np.ndarray  # velocity (3,)
    roll: float
    pitch: float
    yaw: float

@dataclass
class ControlOut:
    roll_cmd: float
    pitch_cmd: float
    yaw_rate_cmd: float
    thrust_cmd: float  # Newtons

@dataclass
class RefOut:
    p_ref: np.ndarray  # (3,)
    yaw_ref: float


# ---------------------------
# Reference program
# ---------------------------

class ReferenceProgram:
    def __init__(self, flight_plan: List[Dict[str, Any]], rng: np.random.Generator):
        self.plan = flight_plan
        self.rng = rng
        self.seg_idx = 0
        self.seg_t = 0.0
        self.seg_hold_t = 0.0
        self.current_target = None
        self.current_yaw_target = None

    def reset(self):
        self.seg_idx = 0
        self.seg_t = 0.0
        self.seg_hold_t = 0.0
        self.current_target = None
        self.current_yaw_target = None

    def step(self, t: float, dt: float, state: QuadState) -> RefOut:
        if self.seg_idx >= len(self.plan):
            # default hover at current position
            return RefOut(p_ref=state.p.copy(), yaw_ref=state.yaw)

        seg = self.plan[self.seg_idx]
        seg_type = seg.get("type")
        seg_dur = float(seg.get("duration_s", 0.0))

        # advance segment time
        self.seg_t += dt
        if self.seg_t > seg_dur:
            self.seg_idx += 1
            self.seg_t = 0.0
            self.seg_hold_t = 0.0
            self.current_target = None
            self.current_yaw_target = None
            return self.step(t, dt, state)

        if seg_type == "hover":
            pos = np.array(seg.get("pos", [0, 0, 1.0]), dtype=np.float64)
            yaw_mode = seg.get("yaw_mode", "hold")
            if yaw_mode == "hold":
                yaw_ref = state.yaw
            else:
                yaw_ref = float(seg.get("yaw", 0.0))
            return RefOut(p_ref=pos, yaw_ref=yaw_ref)

        if seg_type == "random_steps":
            bounds = seg["bounds"]
            hold_s = seg.get("hold_s", [1.0, 3.0])
            yaw_step_prob = float(seg.get("yaw_step_prob", 0.0))
            yaw_step_deg = seg.get("yaw_step_deg", [-45, 45])

            if self.current_target is None:
                self.current_target = self._sample_pos(bounds)
                self.seg_hold_t = 0.0

                if (self.rng.random() < yaw_step_prob) or (self.current_yaw_target is None):
                    d = self.rng.uniform(float(yaw_step_deg[0]), float(yaw_step_deg[1]))
                    self.current_yaw_target = wrap_pi(state.yaw + deg2rad(d))
                else:
                    self.current_yaw_target = state.yaw

                self.current_hold = float(self.rng.uniform(float(hold_s[0]), float(hold_s[1])))

            self.seg_hold_t += dt
            if self.seg_hold_t >= self.current_hold:
                self.current_target = self._sample_pos(bounds)
                self.seg_hold_t = 0.0
                if self.rng.random() < yaw_step_prob:
                    d = self.rng.uniform(float(yaw_step_deg[0]), float(yaw_step_deg[1]))
                    self.current_yaw_target = wrap_pi(state.yaw + deg2rad(d))
                self.current_hold = float(self.rng.uniform(float(hold_s[0]), float(hold_s[1])))

            return RefOut(p_ref=self.current_target.copy(), yaw_ref=float(self.current_yaw_target))

        if seg_type == "circle":
            center = np.array(seg.get("center", [0, 0, 1.0]), dtype=np.float64)
            radius = float(seg.get("radius", 1.0))
            omega = float(seg.get("omega", 0.8))
            yaw_mode = seg.get("yaw_mode", "hold")

            ang = omega * self.seg_t
            p_ref = center + np.array([radius * math.cos(ang), radius * math.sin(ang), 0.0], dtype=np.float64)

            if yaw_mode == "face_velocity":
                # tangent direction (velocity of the reference)
                vx = -radius * omega * math.sin(ang)
                vy =  radius * omega * math.cos(ang)
                yaw_ref = math.atan2(vy, vx)
            else:
                yaw_ref = state.yaw

            return RefOut(p_ref=p_ref, yaw_ref=yaw_ref)

        # fallback: hold position
        return RefOut(p_ref=state.p.copy(), yaw_ref=state.yaw)

    def _sample_pos(self, bounds: Dict[str, List[float]]) -> np.ndarray:
        x = self.rng.uniform(bounds["x"][0], bounds["x"][1])
        y = self.rng.uniform(bounds["y"][0], bounds["y"][1])
        z = self.rng.uniform(bounds["z"][0], bounds["z"][1])
        return np.array([x, y, z], dtype=np.float64)


# ---------------------------
# Cascaded controller
# ---------------------------

class CascadedController:
    def __init__(self, cfg: Dict[str, Any]):
        # Reasonable defaults; you can expose later
        self.kp_pos = float(cfg.get("kp_pos", 2.0))
        self.kd_pos = float(cfg.get("kd_pos", 2.5))
        self.kp_yaw = float(cfg.get("kp_yaw", 3.0))

        self.tilt_max = deg2rad(float(cfg.get("tilt_max_deg", 25.0)))
        self.yaw_rate_max = deg2rad(float(cfg.get("yaw_rate_max_dps", 180.0)))

        self.thrust_min = float(cfg.get("thrust_min", 0.0))
        self.thrust_max = float(cfg.get("thrust_max", 30.0))  # for ~1kg toy; will clamp anyway

        # first-order attitude response time constants (sim side); controller just outputs cmds

    def compute(self, state: QuadState, ref: RefOut, mass: float, g: float) -> ControlOut:
        e_p = ref.p_ref - state.p
        e_v = -state.v

        # Outer loop: desired acceleration in world
        a_star = self.kp_pos * e_p + self.kd_pos * e_v

        # Add gravity compensation target (want thrust to counter gravity)
        # We'll command a_cmd where z includes +g to hover when a_star=0.
        a_cmd = a_star + np.array([0.0, 0.0, g], dtype=np.float64)

        # Map desired accel to roll/pitch given current yaw
        # Small-angle mapping:
        #   phi*   = (a_x*sin(psi) - a_y*cos(psi)) / g
        #   theta* = (a_x*cos(psi) + a_y*sin(psi)) / g
        psi = state.yaw
        ax, ay, az = float(a_cmd[0]), float(a_cmd[1]), float(a_cmd[2])

        roll_cmd = (ax * math.sin(psi) - ay * math.cos(psi)) / max(g, 1e-6)
        pitch_cmd = (ax * math.cos(psi) + ay * math.sin(psi)) / max(g, 1e-6)

        roll_cmd = clamp(roll_cmd, -self.tilt_max, self.tilt_max)
        pitch_cmd = clamp(pitch_cmd, -self.tilt_max, self.tilt_max)

        # Collective thrust (Newton): roughly m * a_z
        thrust_cmd = mass * az
        thrust_cmd = clamp(thrust_cmd, self.thrust_min, self.thrust_max)

        # Yaw: simple P controller producing yaw rate
        e_yaw = wrap_pi(ref.yaw_ref - state.yaw)
        yaw_rate_cmd = clamp(self.kp_yaw * e_yaw, -self.yaw_rate_max, self.yaw_rate_max)

        return ControlOut(roll_cmd=roll_cmd, pitch_cmd=pitch_cmd, yaw_rate_cmd=yaw_rate_cmd, thrust_cmd=thrust_cmd)


# ---------------------------
# Dynamics (toy but drone-like)
# ---------------------------

class ToyQuadDynamics:
    def __init__(self, cfg: Dict[str, Any]):
        self.g = float(cfg.get("gravity", 9.81))
        self.drag = float(cfg.get("drag_coeff", 0.08))

        # First-order attitude tracking (how quickly vehicle achieves commanded tilt)
        self.tau_roll = float(cfg.get("tau_roll", 0.08))
        self.tau_pitch = float(cfg.get("tau_pitch", 0.08))

        # Optional thrust lag (very mild)
        self.tau_thrust = float(cfg.get("tau_thrust", 0.05))
        self._thrust_actual = 0.0

        # Wind / disturbance
        self.wind = np.array(cfg.get("wind", [0.0, 0.0, 0.0]), dtype=np.float64)

    def reset(self):
        self._thrust_actual = 0.0

    def step(self, state: QuadState, u: ControlOut, mass: float, dt: float) -> QuadState:
        # attitude first-order response
        roll = state.roll + (dt / max(self.tau_roll, 1e-6)) * (u.roll_cmd - state.roll)
        pitch = state.pitch + (dt / max(self.tau_pitch, 1e-6)) * (u.pitch_cmd - state.pitch)
        yaw = wrap_pi(state.yaw + u.yaw_rate_cmd * dt)

        # thrust lag
        self._thrust_actual = self._thrust_actual + (dt / max(self.tau_thrust, 1e-6)) * (u.thrust_cmd - self._thrust_actual)
        T = self._thrust_actual

        # world thrust direction
        R = rpy_to_rotmat(roll, pitch, yaw)
        b3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # body +Z
        F = T * (R @ b3)  # world force

        # translational dynamics
        gvec = np.array([0.0, 0.0, -self.g], dtype=np.float64)

        a = (F / max(mass, 1e-6)) + gvec - self.drag * state.v + self.wind

        v = state.v + a * dt
        p = state.p + v * dt

        return QuadState(p=p, v=v, roll=roll, pitch=pitch, yaw=yaw)


# ---------------------------
# Episode runner
# ---------------------------

def sample_init(init_cfg: Dict[str, Any], rng: np.random.Generator) -> QuadState:
    def sample_box(box: Dict[str, List[float]]) -> np.ndarray:
        return np.array([
            rng.uniform(box["x"][0], box["x"][1]),
            rng.uniform(box["y"][0], box["y"][1]),
            rng.uniform(box["z"][0], box["z"][1]),
        ], dtype=np.float64)

    pos_box = init_cfg.get("pos_box", {"x":[0,0], "y":[0,0], "z":[1,1]})
    vel_box = init_cfg.get("vel_box", {"x":[0,0], "y":[0,0], "z":[0,0]})
    yaw_deg_range = init_cfg.get("yaw_deg", [0, 0])

    p = sample_box(pos_box)
    v = sample_box(vel_box)
    yaw = deg2rad(float(rng.uniform(yaw_deg_range[0], yaw_deg_range[1])))

    return QuadState(p=p, v=v, roll=0.0, pitch=0.0, yaw=yaw)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def run_episode(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    run_id = cfg["run_id"]
    seed = int(cfg["seed"])
    dt = float(cfg["dt"])
    duration_s = float(cfg["duration_s"])
    steps = int(math.ceil(duration_s / dt))

    # NEW: real-time throttle toggle
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

    # Controller settings (allow override later)
    ctrl_cfg = cfg.get("controller", {})
    controller = CascadedController(ctrl_cfg)

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
    log_path = os.path.join(out_dir, f"{run_id}_{ts}.csv")

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

    # NEW: wall-clock pacing baseline
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
            u = controller.compute(state, ref, mass=mass, g=gravity)
            # Telemetry/visual helper: normalize thrust for prop spin etc. (0..1)
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

            # advance sim time
            t += dt

            # NEW: Real-time throttle (keeps sim-time fixed, paces wall-clock)
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
        "real_time": real_time,  # NEW: record mode in summary
    }
    return log_path, summary


# ---------------------------
# WebSocket server (one-shot)
# ---------------------------

async def handler(ws):
    msg = await ws.recv()
    print("Received JSON from Unity:")
    print(msg)

    cfg = json.loads(msg)

    # Minimal validation
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

    # Exit after one run
    asyncio.get_running_loop().call_soon(asyncio.get_running_loop().stop)


async def main():
    print(f"ToyQuad server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        # loop intentionally stopped
        pass
    print("Server exiting.")
