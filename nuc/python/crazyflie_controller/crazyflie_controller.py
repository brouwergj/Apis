from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from .controller_dispatch import ControllerDispatch
from .controller_pid import RateContext
from .types import Attitude, AttitudeMode, Control, Quaternion, SensorData, Setpoint, StabMode, State, Vec3


@dataclass
class ControlOut:
    roll_cmd: float
    pitch_cmd: float
    yaw_rate_cmd: float
    thrust_cmd: float


def _base_defaults_cf2() -> Dict[str, Any]:
    return {
        # Attitude filters
        "attitude_lpf_cutoff": 15.0,
        "attitude_lpf_enable": False,
        "attitude_roll_rate_lpf_cutoff": 30.0,
        "attitude_pitch_rate_lpf_cutoff": 30.0,
        "attitude_yaw_rate_lpf_cutoff": 30.0,
        "attitude_rate_lpf_enable": False,
        "yaw_max_delta": 0.0,
        # Attitude rate PID
        "pid_roll_rate_kp": 250.0,
        "pid_roll_rate_ki": 500.0,
        "pid_roll_rate_kd": 2.5,
        "pid_roll_rate_kff": 0.0,
        "pid_roll_rate_i_limit": 33.3,
        "pid_pitch_rate_kp": 250.0,
        "pid_pitch_rate_ki": 500.0,
        "pid_pitch_rate_kd": 2.5,
        "pid_pitch_rate_kff": 0.0,
        "pid_pitch_rate_i_limit": 33.3,
        "pid_yaw_rate_kp": 120.0,
        "pid_yaw_rate_ki": 16.7,
        "pid_yaw_rate_kd": 0.0,
        "pid_yaw_rate_kff": 0.0,
        "pid_yaw_rate_i_limit": 166.7,
        # Attitude PID
        "pid_roll_kp": 6.0,
        "pid_roll_ki": 3.0,
        "pid_roll_kd": 0.0,
        "pid_roll_kff": 0.0,
        "pid_roll_i_limit": 20.0,
        "pid_pitch_kp": 6.0,
        "pid_pitch_ki": 3.0,
        "pid_pitch_kd": 0.0,
        "pid_pitch_kff": 0.0,
        "pid_pitch_i_limit": 20.0,
        "pid_yaw_kp": 6.0,
        "pid_yaw_ki": 1.0,
        "pid_yaw_kd": 0.35,
        "pid_yaw_kff": 0.0,
        "pid_yaw_i_limit": 360.0,
        # Position/velocity PID
        "pid_vel_x_kp": 25.0,
        "pid_vel_x_ki": 1.0,
        "pid_vel_x_kd": 0.0,
        "pid_vel_x_kff": 0.0,
        "pid_vel_y_kp": 25.0,
        "pid_vel_y_ki": 1.0,
        "pid_vel_y_kd": 0.0,
        "pid_vel_y_kff": 0.0,
        "pid_vel_z_kp": 25.0,
        "pid_vel_z_ki": 15.0,
        "pid_vel_z_kd": 0.0,
        "pid_vel_z_kff": 0.0,
        "pid_vel_roll_max": 20.0,
        "pid_vel_pitch_max": 20.0,
        "pid_vel_thrust_base": 36000.0,
        "pid_vel_thrust_min": 20000.0,
        "pid_pos_x_kp": 2.0,
        "pid_pos_x_ki": 0.0,
        "pid_pos_x_kd": 0.0,
        "pid_pos_x_kff": 0.0,
        "pid_pos_y_kp": 2.0,
        "pid_pos_y_ki": 0.0,
        "pid_pos_y_kd": 0.0,
        "pid_pos_y_kff": 0.0,
        "pid_pos_z_kp": 2.0,
        "pid_pos_z_ki": 0.5,
        "pid_pos_z_kd": 0.0,
        "pid_pos_z_kff": 0.0,
        "pid_pos_vel_x_max": 1.0,
        "pid_pos_vel_y_max": 1.0,
        "pid_pos_vel_z_max": 1.0,
        # Filters for position/velocity
        "pid_pos_xy_filt_enable": True,
        "pid_pos_xy_filt_cutoff": 20.0,
        "pid_pos_z_filt_enable": True,
        "pid_pos_z_filt_cutoff": 20.0,
        "pid_vel_xy_filt_enable": True,
        "pid_vel_xy_filt_cutoff": 20.0,
        "pid_vel_z_filt_enable": True,
        "pid_vel_z_filt_cutoff": 20.0,
        # Overheads and scaling
        "pid_rp_limit_overhead": 1.10,
        "pid_vel_max_overhead": 1.10,
        "pid_vel_thrust_scale": 1000.0,
    }


def build_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = _base_defaults_cf2()
    overrides = cfg.get("overrides", {}) if isinstance(cfg, dict) else {}
    for key, val in overrides.items():
        if key in base:
            base[key] = val
    # Allow flat overrides too
    if isinstance(cfg, dict):
        for key, val in cfg.items():
            if key in base:
                base[key] = val
    return base


class CrazyflieController:
    def __init__(self, cfg: Dict[str, Any], dt: float) -> None:
        self.cfg = build_config(cfg or {})
        self.dt = dt
        self.rate_context = self._make_rate_context(dt)
        self.dispatch = ControllerDispatch(cfg.get("cf_controller", "pid"), self.cfg, self.rate_context)
        self.stabilizer_step = 0
        self._prev_state = None
        self._prev_velocity = None

        self._thrust_base = float(self.cfg.get("pid_vel_thrust_base", 36000.0))
        self._thrust_scale_n = cfg.get("thrust_to_newton", None)

    def _make_rate_context(self, dt: float) -> RateContext:
        main_rate_hz = max(1, int(round(1.0 / max(dt, 1e-6))))
        attitude_rate_hz = min(500, main_rate_hz)
        position_rate_hz = min(100, main_rate_hz)
        return RateContext(
            main_rate_hz=main_rate_hz,
            attitude_rate_hz=attitude_rate_hz,
            position_rate_hz=position_rate_hz,
        )

    def _make_state(self, state_sim, acc_sim, gyro_sim) -> State:
        roll_deg = math.degrees(state_sim.roll)
        pitch_deg = -math.degrees(state_sim.pitch)
        yaw_deg = math.degrees(state_sim.yaw)

        attitude = Attitude(roll_deg, pitch_deg, yaw_deg)
        attitude_quat = Quaternion(0.0, 0.0, 0.0, 1.0)

        position = Vec3(float(state_sim.p[0]), float(state_sim.p[1]), float(state_sim.p[2]))
        velocity = Vec3(float(state_sim.v[0]), float(state_sim.v[1]), float(state_sim.v[2]))

        acc = Vec3(acc_sim[0], acc_sim[1], acc_sim[2])

        return State(attitude=attitude, attitude_quat=attitude_quat, position=position, velocity=velocity, acc=acc)

    def _make_sensors(self, gyro_sim, acc_sim) -> SensorData:
        gyro = Vec3(gyro_sim[0], gyro_sim[1], gyro_sim[2])
        acc = Vec3(acc_sim[0], acc_sim[1], acc_sim[2])
        return SensorData(gyro=gyro, acc=acc)

    def _make_setpoint(self, ref) -> Setpoint:
        yaw_deg = math.degrees(ref.yaw_ref)
        attitude = Attitude(0.0, 0.0, yaw_deg)
        attitude_rate = Attitude(0.0, 0.0, 0.0)
        attitude_quat = Quaternion(0.0, 0.0, 0.0, 1.0)
        position = Vec3(float(ref.p_ref[0]), float(ref.p_ref[1]), float(ref.p_ref[2]))
        velocity = Vec3(0.0, 0.0, 0.0)
        acceleration = Vec3(0.0, 0.0, 0.0)
        jerk = Vec3(0.0, 0.0, 0.0)

        mode = AttitudeMode(
            x=StabMode.ABS,
            y=StabMode.ABS,
            z=StabMode.ABS,
            roll=StabMode.ABS,
            pitch=StabMode.ABS,
            yaw=StabMode.ABS,
            quat=StabMode.DISABLE,
        )

        return Setpoint(
            attitude=attitude,
            attitude_rate=attitude_rate,
            attitude_quat=attitude_quat,
            thrust=0.0,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            velocity_body=False,
            mode=mode,
        )

    def _estimate_gyro(self, state_sim) -> tuple[float, float, float]:
        if self._prev_state is None:
            return 0.0, 0.0, 0.0
        dt = max(self.dt, 1e-6)
        droll = self._wrap_rad(state_sim.roll - self._prev_state.roll)
        dpitch = self._wrap_rad(state_sim.pitch - self._prev_state.pitch)
        dyaw = self._wrap_rad(state_sim.yaw - self._prev_state.yaw)
        return (
            math.degrees(droll / dt),
            math.degrees(dpitch / dt),
            math.degrees(dyaw / dt),
        )

    def _estimate_acc(self, state_sim, g: float) -> tuple[float, float, float]:
        if self._prev_velocity is None:
            return 0.0, 0.0, 0.0
        dt = max(self.dt, 1e-6)
        ax = (state_sim.v[0] - self._prev_velocity[0]) / dt
        ay = (state_sim.v[1] - self._prev_velocity[1]) / dt
        az = (state_sim.v[2] - self._prev_velocity[2]) / dt
        g_val = max(g, 1e-6)
        return ax / g_val, ay / g_val, az / g_val

    def _wrap_rad(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _thrust_to_newton(self, thrust_cmd: float, mass: float, g: float) -> float:
        if self._thrust_scale_n is not None:
            return thrust_cmd * float(self._thrust_scale_n)
        base = max(self._thrust_base, 1e-6)
        return (thrust_cmd / base) * mass * g

    def compute(self, state_sim, ref, mass: float, g: float, dt: float) -> ControlOut:
        if abs(dt - self.dt) > 1e-9:
            self.dt = dt
            self.rate_context = self._make_rate_context(dt)

        gyro_sim = self._estimate_gyro(state_sim)
        acc_sim = self._estimate_acc(state_sim, g)

        state = self._make_state(state_sim, acc_sim, gyro_sim)
        sensors = self._make_sensors(gyro_sim, acc_sim)
        setpoint = self._make_setpoint(ref)

        control = Control(roll=0, pitch=0, yaw=0, thrust=0.0)
        self.dispatch.update(control, setpoint, sensors, state, self.stabilizer_step)
        self.stabilizer_step += 1

        self._prev_state = state_sim
        self._prev_velocity = state_sim.v.copy()

        roll_cmd = math.radians(self.dispatch.pid.attitude_desired.roll)
        pitch_cmd = -math.radians(self.dispatch.pid.attitude_desired.pitch)
        yaw_rate_cmd = math.radians(self.dispatch.pid.rate_desired.yaw)
        thrust_cmd = self._thrust_to_newton(self.dispatch.pid.actuator_thrust, mass, g)

        return ControlOut(
            roll_cmd=roll_cmd,
            pitch_cmd=pitch_cmd,
            yaw_rate_cmd=yaw_rate_cmd,
            thrust_cmd=thrust_cmd,
        )
