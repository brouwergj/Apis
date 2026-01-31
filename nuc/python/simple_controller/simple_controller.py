# simple_controller/simple_controller.py
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


# ---------- tiny standalone utils (portable) ----------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0

def wrap_pi(a: float) -> float:
    # maps to (-pi, pi]
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------- return type (duck-typed by controller.py) ----------

@dataclass
class ControlOut:
    roll_cmd: float
    pitch_cmd: float
    yaw_rate_cmd: float
    thrust_cmd: float


class SimpleController:
    """
    Teacher 1: PD position -> accel, gravity comp -> thrust,
    small-angle mapping -> roll/pitch, P yaw -> yaw rate,
    then saturations + per-step slew limits.
    """

    def __init__(self, cfg: dict):
        # Gains
        self.kp_pos = float(cfg.get("kp_pos", 1.8))
        self.kd_pos = float(cfg.get("kd_pos", 2.2))
        self.kp_yaw = float(cfg.get("kp_yaw", 2.5))

        # Saturations
        self.tilt_max = deg2rad(float(cfg.get("tilt_max_deg", 20.0)))
        self.yaw_rate_max = deg2rad(float(cfg.get("yaw_rate_max_dps", 120.0)))
        self.thrust_min = float(cfg.get("thrust_min", 0.0))
        self.thrust_max = float(cfg.get("thrust_max", 30.0))

        # Per-step slew limits (discrete-time)
        self.roll_step_max = deg2rad(float(cfg.get("roll_step_max_deg", 2.0)))
        self.pitch_step_max = deg2rad(float(cfg.get("pitch_step_max_deg", 2.0)))
        self.yaw_rate_step_max = deg2rad(float(cfg.get("yaw_rate_step_max_dps", 15.0)))
        self.thrust_step_max = float(cfg.get("thrust_step_max", 1.0))

        self._prev_u: ControlOut | None = None

    def reset(self):
        self._prev_u = None

    def _slew(self, desired: ControlOut) -> ControlOut:
        if self._prev_u is None:
            self._prev_u = desired
            return desired

        def step_limit(x, xprev, step_max):
            dx = clamp(x - xprev, -step_max, step_max)
            return xprev + dx

        out = ControlOut(
            roll_cmd=step_limit(desired.roll_cmd, self._prev_u.roll_cmd, self.roll_step_max),
            pitch_cmd=step_limit(desired.pitch_cmd, self._prev_u.pitch_cmd, self.pitch_step_max),
            yaw_rate_cmd=step_limit(desired.yaw_rate_cmd, self._prev_u.yaw_rate_cmd, self.yaw_rate_step_max),
            thrust_cmd=step_limit(desired.thrust_cmd, self._prev_u.thrust_cmd, self.thrust_step_max),
        )
        self._prev_u = out
        return out

    def compute(self, state, ref, mass: float, g: float) -> ControlOut:
        """
        Expects duck-typed inputs:
          state.p: np array shape (3,)
          state.v: np array shape (3,)
          state.yaw: float (rad)
          ref.p_ref: np array shape (3,)
          ref.yaw_ref: float (rad)
        """

        # Position PD in world frame
        e_p = ref.p_ref - state.p
        e_v = -state.v
        a_star = self.kp_pos * e_p + self.kd_pos * e_v

        # Gravity compensation
        a_cmd = a_star + np.array([0.0, 0.0, g], dtype=np.float64)

        # Small-angle mapping accel -> roll/pitch using current yaw
        psi = float(state.yaw)
        ax, ay, az = float(a_cmd[0]), float(a_cmd[1]), float(a_cmd[2])

        roll_cmd = (ax * math.sin(psi) - ay * math.cos(psi)) / max(g, 1e-6)
        pitch_cmd = (ax * math.cos(psi) + ay * math.sin(psi)) / max(g, 1e-6)

        # Tilt saturation
        roll_cmd = clamp(roll_cmd, -self.tilt_max, self.tilt_max)
        pitch_cmd = clamp(pitch_cmd, -self.tilt_max, self.tilt_max)

        # Thrust saturation
        thrust_cmd = clamp(mass * az, self.thrust_min, self.thrust_max)

        # Yaw-rate P + saturation
        e_yaw = wrap_pi(float(ref.yaw_ref) - float(state.yaw))
        yaw_rate_cmd = clamp(self.kp_yaw * e_yaw, -self.yaw_rate_max, self.yaw_rate_max)

        desired = ControlOut(roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd)
        return self._slew(desired)
