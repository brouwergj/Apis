from __future__ import annotations

import math
from dataclasses import dataclass

from .pid import PidObject, constrain
from .types import Attitude, Setpoint, State, StabMode, Vec3


UINT16_MAX = 65535


@dataclass
class PidAxis:
    pid: PidObject
    previous_mode: StabMode = StabMode.DISABLE
    setpoint: float = 0.0
    output: float = 0.0


class PositionController:
    def __init__(self, config: dict, position_rate_hz: float) -> None:
        self._config = config
        self._dt = 1.0 / max(position_rate_hz, 1.0)
        self._pos_filt_enable = bool(config.get("pid_pos_xy_filt_enable", True))
        self._vel_filt_enable = bool(config.get("pid_vel_xy_filt_enable", True))
        self._pos_filt_cutoff = float(config.get("pid_pos_xy_filt_cutoff", 20.0))
        self._vel_filt_cutoff = float(config.get("pid_vel_xy_filt_cutoff", 20.0))
        self._pos_z_filt_enable = bool(config.get("pid_pos_z_filt_enable", True))
        self._vel_z_filt_enable = bool(config.get("pid_vel_z_filt_enable", True))
        self._pos_z_filt_cutoff = float(config.get("pid_pos_z_filt_cutoff", 20.0))
        self._vel_z_filt_cutoff = float(config.get("pid_vel_z_filt_cutoff", 20.0))

        self.pid_vx = PidAxis(PidObject())
        self.pid_vy = PidAxis(PidObject())
        self.pid_vz = PidAxis(PidObject())
        self.pid_x = PidAxis(PidObject())
        self.pid_y = PidAxis(PidObject())
        self.pid_z = PidAxis(PidObject())

        self.r_limit = float(config.get("pid_vel_roll_max", 20.0))
        self.p_limit = float(config.get("pid_vel_pitch_max", 20.0))
        self.rp_limit_overhead = float(config.get("pid_rp_limit_overhead", 1.10))

        self.x_vel_max = float(config.get("pid_pos_vel_x_max", 1.0))
        self.y_vel_max = float(config.get("pid_pos_vel_y_max", 1.0))
        self.z_vel_max = float(config.get("pid_pos_vel_z_max", 1.0))
        self.vel_max_overhead = float(config.get("pid_vel_max_overhead", 1.10))

        self.thrust_base = float(config.get("pid_vel_thrust_base", 36000.0))
        self.thrust_min = float(config.get("pid_vel_thrust_min", 20000.0))

        self.thrust_scale = float(config.get("pid_vel_thrust_scale", 1000.0))

        self.init(position_rate_hz)

    def init(self, position_rate_hz: float) -> None:
        dt = 1.0 / max(position_rate_hz, 1.0)
        self._dt = dt

        self.pid_x.pid.init(
            0.0,
            float(self._get_cfg("pid_pos_x_kp", 2.0)),
            float(self._get_cfg("pid_pos_x_ki", 0.0)),
            float(self._get_cfg("pid_pos_x_kd", 0.0)),
            float(self._get_cfg("pid_pos_x_kff", 0.0)),
            dt,
            position_rate_hz,
            self._pos_filt_cutoff,
            self._pos_filt_enable,
        )
        self.pid_y.pid.init(
            0.0,
            float(self._get_cfg("pid_pos_y_kp", 2.0)),
            float(self._get_cfg("pid_pos_y_ki", 0.0)),
            float(self._get_cfg("pid_pos_y_kd", 0.0)),
            float(self._get_cfg("pid_pos_y_kff", 0.0)),
            dt,
            position_rate_hz,
            self._pos_filt_cutoff,
            self._pos_filt_enable,
        )
        self.pid_z.pid.init(
            0.0,
            float(self._get_cfg("pid_pos_z_kp", 2.0)),
            float(self._get_cfg("pid_pos_z_ki", 0.5)),
            float(self._get_cfg("pid_pos_z_kd", 0.0)),
            float(self._get_cfg("pid_pos_z_kff", 0.0)),
            dt,
            position_rate_hz,
            self._pos_z_filt_cutoff,
            self._pos_z_filt_enable,
        )

        self.pid_vx.pid.init(
            0.0,
            float(self._get_cfg("pid_vel_x_kp", 25.0)),
            float(self._get_cfg("pid_vel_x_ki", 1.0)),
            float(self._get_cfg("pid_vel_x_kd", 0.0)),
            float(self._get_cfg("pid_vel_x_kff", 0.0)),
            dt,
            position_rate_hz,
            self._vel_filt_cutoff,
            self._vel_filt_enable,
        )
        self.pid_vy.pid.init(
            0.0,
            float(self._get_cfg("pid_vel_y_kp", 25.0)),
            float(self._get_cfg("pid_vel_y_ki", 1.0)),
            float(self._get_cfg("pid_vel_y_kd", 0.0)),
            float(self._get_cfg("pid_vel_y_kff", 0.0)),
            dt,
            position_rate_hz,
            self._vel_filt_cutoff,
            self._vel_filt_enable,
        )
        self.pid_vz.pid.init(
            0.0,
            float(self._get_cfg("pid_vel_z_kp", 25.0)),
            float(self._get_cfg("pid_vel_z_ki", 15.0)),
            float(self._get_cfg("pid_vel_z_kd", 0.0)),
            float(self._get_cfg("pid_vel_z_kff", 0.0)),
            dt,
            position_rate_hz,
            self._vel_z_filt_cutoff,
            self._vel_z_filt_enable,
        )

    def _get_cfg(self, key: str, default: float) -> float:
        if key in self._config:
            return float(self._config[key])
        return default

    def _run_pid(self, input_val: float, axis: PidAxis, setpoint: float) -> float:
        axis.setpoint = setpoint
        axis.pid.set_desired(axis.setpoint)
        axis.output = axis.pid.update(input_val, is_yaw_angle=False)
        return axis.output

    def position_controller(self, thrust_out: list, attitude_out: Attitude, setpoint: Setpoint, state: State) -> None:
        self.pid_x.pid.output_limit = self.x_vel_max * self.vel_max_overhead
        self.pid_y.pid.output_limit = self.y_vel_max * self.vel_max_overhead
        self.pid_z.pid.output_limit = max(self.z_vel_max, 0.5) * self.vel_max_overhead

        yaw_rad = math.radians(state.attitude.yaw)
        cosyaw = math.cos(yaw_rad)
        sinyaw = math.sin(yaw_rad)

        setp_body_x = setpoint.position.x * cosyaw + setpoint.position.y * sinyaw
        setp_body_y = -setpoint.position.x * sinyaw + setpoint.position.y * cosyaw

        state_body_x = state.position.x * cosyaw + state.position.y * sinyaw
        state_body_y = -state.position.x * sinyaw + state.position.y * cosyaw

        global_vx = setpoint.velocity.x
        global_vy = setpoint.velocity.y

        setpoint_velocity = Vec3(setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z)

        if setpoint.mode.x == StabMode.ABS:
            setpoint_velocity.x = self._run_pid(state_body_x, self.pid_x, setp_body_x)
        elif not setpoint.velocity_body:
            setpoint_velocity.x = global_vx * cosyaw + global_vy * sinyaw

        if setpoint.mode.y == StabMode.ABS:
            setpoint_velocity.y = self._run_pid(state_body_y, self.pid_y, setp_body_y)
        elif not setpoint.velocity_body:
            setpoint_velocity.y = global_vy * cosyaw - global_vx * sinyaw

        if setpoint.mode.z == StabMode.ABS:
            setpoint_velocity.z = self._run_pid(state.position.z, self.pid_z, setpoint.position.z)

        self.velocity_controller(thrust_out, attitude_out, setpoint_velocity, state)

    def velocity_controller(self, thrust_out: list, attitude_out: Attitude, setpoint_velocity: Vec3, state: State) -> None:
        self.pid_vx.pid.output_limit = self.p_limit * self.rp_limit_overhead
        self.pid_vy.pid.output_limit = self.r_limit * self.rp_limit_overhead
        self.pid_vz.pid.output_limit = (UINT16_MAX / 2 / self.thrust_scale)

        yaw_rad = math.radians(state.attitude.yaw)
        cosyaw = math.cos(yaw_rad)
        sinyaw = math.sin(yaw_rad)
        state_body_vx = state.velocity.x * cosyaw + state.velocity.y * sinyaw
        state_body_vy = -state.velocity.x * sinyaw + state.velocity.y * cosyaw

        attitude_out.pitch = -self._run_pid(state_body_vx, self.pid_vx, setpoint_velocity.x)
        attitude_out.roll = -self._run_pid(state_body_vy, self.pid_vy, setpoint_velocity.y)

        attitude_out.roll = constrain(attitude_out.roll, -self.r_limit, self.r_limit)
        attitude_out.pitch = constrain(attitude_out.pitch, -self.p_limit, self.p_limit)

        thrust_raw = self._run_pid(state.velocity.z, self.pid_vz, setpoint_velocity.z)
        thrust = thrust_raw * self.thrust_scale + self.thrust_base
        if thrust < self.thrust_min:
            thrust = self.thrust_min
        thrust = constrain(thrust, 0.0, float(UINT16_MAX))
        thrust_out[0] = thrust

    def reset_all_pid(self, x_actual: float, y_actual: float, z_actual: float) -> None:
        self.pid_x.pid.reset(x_actual)
        self.pid_y.pid.reset(y_actual)
        self.pid_z.pid.reset(z_actual)
        self.pid_vx.pid.reset(0.0)
        self.pid_vy.pid.reset(0.0)
        self.pid_vz.pid.reset(0.0)

    def reset_all_filters(self, position_rate_hz: float) -> None:
        self.pid_x.pid.reset_filter(position_rate_hz, self._pos_filt_cutoff, self._pos_filt_enable)
        self.pid_y.pid.reset_filter(position_rate_hz, self._pos_filt_cutoff, self._pos_filt_enable)
        self.pid_z.pid.reset_filter(position_rate_hz, self._pos_z_filt_cutoff, self._pos_z_filt_enable)
        self.pid_vx.pid.reset_filter(position_rate_hz, self._vel_filt_cutoff, self._vel_filt_enable)
        self.pid_vy.pid.reset_filter(position_rate_hz, self._vel_filt_cutoff, self._vel_filt_enable)
        self.pid_vz.pid.reset_filter(position_rate_hz, self._vel_z_filt_cutoff, self._vel_z_filt_enable)
