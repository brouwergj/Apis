from __future__ import annotations

import math
from dataclasses import dataclass

from .attitude_controller import AttitudeController
from .position_controller import PositionController
from .types import Attitude, Control, SensorData, Setpoint, StabMode, State


@dataclass
class RateContext:
    main_rate_hz: int
    attitude_rate_hz: int
    position_rate_hz: int

    def do_execute(self, rate_hz: int, step: int) -> bool:
        if rate_hz <= 0:
            return False
        denom = max(1, int(round(self.main_rate_hz / float(rate_hz))))
        return (step % denom) == 0


def cap_angle(angle_deg: float) -> float:
    result = angle_deg
    while result > 180.0:
        result -= 360.0
    while result < -180.0:
        result += 360.0
    return result


def quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)


class ControllerPID:
    def __init__(self, config: dict, rate_context: RateContext) -> None:
        self.rate_context = rate_context
        self.attitude_controller = AttitudeController(
            config,
            update_dt=1.0 / max(rate_context.attitude_rate_hz, 1),
            rate_hz=rate_context.attitude_rate_hz,
        )
        self.position_controller = PositionController(config, rate_context.position_rate_hz)

        self.attitude_desired = Attitude(0.0, 0.0, 0.0)
        self.rate_desired = Attitude(0.0, 0.0, 0.0)
        self.actuator_thrust = 0.0

    def update(
        self,
        control: Control,
        setpoint: Setpoint,
        sensors: SensorData,
        state: State,
        stabilizer_step: int,
    ) -> None:
        control.control_mode = "legacy"

        if self.rate_context.do_execute(self.rate_context.attitude_rate_hz, stabilizer_step):
            if setpoint.mode.yaw == StabMode.VELOCITY:
                self.attitude_desired.yaw = cap_angle(
                    self.attitude_desired.yaw + setpoint.attitude_rate.yaw * (1.0 / max(self.rate_context.attitude_rate_hz, 1))
                )
                yaw_max_delta = self.attitude_controller.get_yaw_max_delta()
                if yaw_max_delta != 0.0:
                    delta = cap_angle(self.attitude_desired.yaw - state.attitude.yaw)
                    if delta > yaw_max_delta:
                        self.attitude_desired.yaw = state.attitude.yaw + yaw_max_delta
                    elif delta < -yaw_max_delta:
                        self.attitude_desired.yaw = state.attitude.yaw - yaw_max_delta
            elif setpoint.mode.yaw == StabMode.ABS:
                self.attitude_desired.yaw = setpoint.attitude.yaw
            elif setpoint.mode.quat == StabMode.ABS:
                self.attitude_desired.yaw = quat_to_yaw_deg(
                    setpoint.attitude_quat.x,
                    setpoint.attitude_quat.y,
                    setpoint.attitude_quat.z,
                    setpoint.attitude_quat.w,
                )
            self.attitude_desired.yaw = cap_angle(self.attitude_desired.yaw)

        if self.rate_context.do_execute(self.rate_context.position_rate_hz, stabilizer_step):
            thrust_holder = [self.actuator_thrust]
            self.position_controller.position_controller(thrust_holder, self.attitude_desired, setpoint, state)
            self.actuator_thrust = thrust_holder[0]

        if self.rate_context.do_execute(self.rate_context.attitude_rate_hz, stabilizer_step):
            if setpoint.mode.z == StabMode.DISABLE:
                self.actuator_thrust = setpoint.thrust
            if setpoint.mode.x == StabMode.DISABLE or setpoint.mode.y == StabMode.DISABLE:
                self.attitude_desired.roll = setpoint.attitude.roll
                self.attitude_desired.pitch = setpoint.attitude.pitch

            roll_rate_des, pitch_rate_des, yaw_rate_des = self.attitude_controller.correct_attitude_pid(
                state.attitude.roll,
                state.attitude.pitch,
                state.attitude.yaw,
                self.attitude_desired.roll,
                self.attitude_desired.pitch,
                self.attitude_desired.yaw,
            )
            self.rate_desired.roll = roll_rate_des
            self.rate_desired.pitch = pitch_rate_des
            self.rate_desired.yaw = yaw_rate_des

            if setpoint.mode.roll == StabMode.VELOCITY:
                self.rate_desired.roll = setpoint.attitude_rate.roll
                self.attitude_controller.reset_roll_attitude_pid(state.attitude.roll)
            if setpoint.mode.pitch == StabMode.VELOCITY:
                self.rate_desired.pitch = setpoint.attitude_rate.pitch
                self.attitude_controller.reset_pitch_attitude_pid(state.attitude.pitch)

            self.attitude_controller.correct_rate_pid(
                sensors.gyro.x,
                -sensors.gyro.y,
                sensors.gyro.z,
                self.rate_desired.roll,
                self.rate_desired.pitch,
                self.rate_desired.yaw,
            )

            roll_out, pitch_out, yaw_out = self.attitude_controller.get_actuator_output()
            control.roll = roll_out
            control.pitch = pitch_out
            control.yaw = -yaw_out

        control.thrust = self.actuator_thrust

        if control.thrust == 0:
            control.thrust = 0.0
            control.roll = 0
            control.pitch = 0
            control.yaw = 0

            self.attitude_controller.reset_all_pid(
                state.attitude.roll, state.attitude.pitch, state.attitude.yaw
            )
            self.position_controller.reset_all_pid(
                state.position.x, state.position.y, state.position.z
            )

            self.attitude_desired.yaw = state.attitude.yaw
