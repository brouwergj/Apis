from __future__ import annotations

from dataclasses import dataclass

from .pid import PidObject


def saturate_signed_int16(value: float) -> int:
    if value > 32767:
        return 32767
    if value < -32767:
        return -32767
    return int(value)


@dataclass
class AttitudeOutputs:
    roll: int = 0
    pitch: int = 0
    yaw: int = 0


class AttitudeController:
    def __init__(self, config: dict, update_dt: float, rate_hz: float) -> None:
        self._att_filt_enable = bool(config.get("attitude_lpf_enable", False))
        self._rate_filt_enable = bool(config.get("attitude_rate_lpf_enable", False))
        self._att_filt_cutoff = float(config.get("attitude_lpf_cutoff", 15.0))
        self._omx_filt_cutoff = float(config.get("attitude_roll_rate_lpf_cutoff", 30.0))
        self._omy_filt_cutoff = float(config.get("attitude_pitch_rate_lpf_cutoff", 30.0))
        self._omz_filt_cutoff = float(config.get("attitude_yaw_rate_lpf_cutoff", 30.0))
        self._yaw_max_delta = float(config.get("yaw_max_delta", 0.0))
        self._filter_all = bool(config.get("pid_filter_all", False))

        self.pid_roll_rate = PidObject()
        self.pid_pitch_rate = PidObject()
        self.pid_yaw_rate = PidObject()
        self.pid_roll = PidObject()
        self.pid_pitch = PidObject()
        self.pid_yaw = PidObject()

        self.outputs = AttitudeOutputs()
        self._is_init = False

        self.init(update_dt, rate_hz, config)

    def init(self, update_dt: float, rate_hz: float, config: dict) -> None:
        if self._is_init:
            return

        self.pid_roll_rate.init(
            0.0,
            float(config.get("pid_roll_rate_kp", 250.0)),
            float(config.get("pid_roll_rate_ki", 500.0)),
            float(config.get("pid_roll_rate_kd", 2.5)),
            float(config.get("pid_roll_rate_kff", 0.0)),
            update_dt,
            rate_hz,
            self._omx_filt_cutoff,
            self._rate_filt_enable,
        )
        self.pid_pitch_rate.init(
            0.0,
            float(config.get("pid_pitch_rate_kp", 250.0)),
            float(config.get("pid_pitch_rate_ki", 500.0)),
            float(config.get("pid_pitch_rate_kd", 2.5)),
            float(config.get("pid_pitch_rate_kff", 0.0)),
            update_dt,
            rate_hz,
            self._omy_filt_cutoff,
            self._rate_filt_enable,
        )
        self.pid_yaw_rate.init(
            0.0,
            float(config.get("pid_yaw_rate_kp", 120.0)),
            float(config.get("pid_yaw_rate_ki", 16.7)),
            float(config.get("pid_yaw_rate_kd", 0.0)),
            float(config.get("pid_yaw_rate_kff", 0.0)),
            update_dt,
            rate_hz,
            self._omz_filt_cutoff,
            self._rate_filt_enable,
        )

        self.pid_roll_rate.set_integral_limit(float(config.get("pid_roll_rate_i_limit", 33.3)))
        self.pid_pitch_rate.set_integral_limit(float(config.get("pid_pitch_rate_i_limit", 33.3)))
        self.pid_yaw_rate.set_integral_limit(float(config.get("pid_yaw_rate_i_limit", 166.7)))

        self.pid_roll.init(
            0.0,
            float(config.get("pid_roll_kp", 6.0)),
            float(config.get("pid_roll_ki", 3.0)),
            float(config.get("pid_roll_kd", 0.0)),
            float(config.get("pid_roll_kff", 0.0)),
            update_dt,
            rate_hz,
            self._att_filt_cutoff,
            self._att_filt_enable,
        )
        self.pid_pitch.init(
            0.0,
            float(config.get("pid_pitch_kp", 6.0)),
            float(config.get("pid_pitch_ki", 3.0)),
            float(config.get("pid_pitch_kd", 0.0)),
            float(config.get("pid_pitch_kff", 0.0)),
            update_dt,
            rate_hz,
            self._att_filt_cutoff,
            self._att_filt_enable,
        )
        self.pid_yaw.init(
            0.0,
            float(config.get("pid_yaw_kp", 6.0)),
            float(config.get("pid_yaw_ki", 1.0)),
            float(config.get("pid_yaw_kd", 0.35)),
            float(config.get("pid_yaw_kff", 0.0)),
            update_dt,
            rate_hz,
            self._att_filt_cutoff,
            self._att_filt_enable,
        )

        self.pid_roll.set_integral_limit(float(config.get("pid_roll_i_limit", 20.0)))
        self.pid_pitch.set_integral_limit(float(config.get("pid_pitch_i_limit", 20.0)))
        self.pid_yaw.set_integral_limit(float(config.get("pid_yaw_i_limit", 360.0)))

        self._is_init = True

    def correct_attitude_pid(
        self,
        roll_actual: float,
        pitch_actual: float,
        yaw_actual: float,
        roll_desired: float,
        pitch_desired: float,
        yaw_desired: float,
    ) -> tuple[float, float, float]:
        self.pid_roll.set_desired(roll_desired)
        roll_rate_desired = self.pid_roll.update(roll_actual, is_yaw_angle=False, filter_all=self._filter_all)

        self.pid_pitch.set_desired(pitch_desired)
        pitch_rate_desired = self.pid_pitch.update(pitch_actual, is_yaw_angle=False, filter_all=self._filter_all)

        self.pid_yaw.set_desired(yaw_desired)
        yaw_rate_desired = self.pid_yaw.update(yaw_actual, is_yaw_angle=True, filter_all=self._filter_all)

        return roll_rate_desired, pitch_rate_desired, yaw_rate_desired

    def correct_rate_pid(
        self,
        roll_rate_actual: float,
        pitch_rate_actual: float,
        yaw_rate_actual: float,
        roll_rate_desired: float,
        pitch_rate_desired: float,
        yaw_rate_desired: float,
    ) -> None:
        self.pid_roll_rate.set_desired(roll_rate_desired)
        roll_out = self.pid_roll_rate.update(roll_rate_actual, is_yaw_angle=False, filter_all=self._filter_all)
        self.outputs.roll = saturate_signed_int16(roll_out)

        self.pid_pitch_rate.set_desired(pitch_rate_desired)
        pitch_out = self.pid_pitch_rate.update(pitch_rate_actual, is_yaw_angle=False, filter_all=self._filter_all)
        self.outputs.pitch = saturate_signed_int16(pitch_out)

        self.pid_yaw_rate.set_desired(yaw_rate_desired)
        yaw_out = self.pid_yaw_rate.update(yaw_rate_actual, is_yaw_angle=False, filter_all=self._filter_all)
        self.outputs.yaw = saturate_signed_int16(yaw_out)

    def reset_roll_attitude_pid(self, roll_actual: float) -> None:
        self.pid_roll.reset(roll_actual)

    def reset_pitch_attitude_pid(self, pitch_actual: float) -> None:
        self.pid_pitch.reset(pitch_actual)

    def reset_all_pid(self, roll_actual: float, pitch_actual: float, yaw_actual: float) -> None:
        self.pid_roll.reset(roll_actual)
        self.pid_pitch.reset(pitch_actual)
        self.pid_yaw.reset(yaw_actual)
        self.pid_roll_rate.reset(0.0)
        self.pid_pitch_rate.reset(0.0)
        self.pid_yaw_rate.reset(0.0)

    def get_actuator_output(self) -> tuple[int, int, int]:
        return self.outputs.roll, self.outputs.pitch, self.outputs.yaw

    def get_yaw_max_delta(self) -> float:
        return self._yaw_max_delta
