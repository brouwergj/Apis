from __future__ import annotations

import math
from dataclasses import dataclass, field

from .filter import Lpf2pData


DEFAULT_PID_INTEGRATION_LIMIT = 5000.0
DEFAULT_PID_OUTPUT_LIMIT = 0.0


def constrain(value: float, min_val: float, max_val: float) -> float:
    return min(max_val, max(min_val, value))


@dataclass
class PidObject:
    desired: float = 0.0
    error: float = 0.0
    prev_measured: float = 0.0
    integ: float = 0.0
    deriv: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    kff: float = 0.0
    out_p: float = 0.0
    out_i: float = 0.0
    out_d: float = 0.0
    out_ff: float = 0.0
    i_limit: float = DEFAULT_PID_INTEGRATION_LIMIT
    output_limit: float = DEFAULT_PID_OUTPUT_LIMIT
    dt: float = 0.0
    d_filter: Lpf2pData = field(default_factory=Lpf2pData)
    enable_d_filter: bool = False

    def init(self, desired: float, kp: float, ki: float, kd: float, kff: float,
             dt: float, sampling_rate: float, cutoff_freq: float, enable_d_filter: bool) -> None:
        self.error = 0.0
        self.prev_measured = 0.0
        self.integ = 0.0
        self.deriv = 0.0
        self.desired = desired
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        self.i_limit = DEFAULT_PID_INTEGRATION_LIMIT
        self.output_limit = DEFAULT_PID_OUTPUT_LIMIT
        self.dt = dt
        self.enable_d_filter = enable_d_filter
        if self.enable_d_filter:
            self.d_filter.init(sampling_rate, cutoff_freq)

    def update(self, measured: float, is_yaw_angle: bool, filter_all: bool = False) -> float:
        output = 0.0

        self.error = self.desired - measured
        if is_yaw_angle:
            if self.error > 180.0:
                self.error -= 360.0
            elif self.error < -180.0:
                self.error += 360.0

        self.out_p = self.kp * self.error
        output += self.out_p

        delta = -(measured - self.prev_measured)
        if is_yaw_angle:
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0

        if self.enable_d_filter and not filter_all:
            self.deriv = self.d_filter.apply(delta / max(self.dt, 1e-6))
        else:
            self.deriv = delta / max(self.dt, 1e-6)

        if math.isnan(self.deriv):
            self.deriv = 0.0

        self.out_d = self.kd * self.deriv
        output += self.out_d

        self.integ += self.error * self.dt
        if self.i_limit != 0.0:
            self.integ = constrain(self.integ, -self.i_limit, self.i_limit)

        self.out_i = self.ki * self.integ
        output += self.out_i

        self.out_ff = self.kff * self.desired
        output += self.out_ff

        if filter_all and self.enable_d_filter:
            output = self.d_filter.apply(output)
            if math.isnan(output):
                output = 0.0

        if self.output_limit != 0.0:
            output = constrain(output, -self.output_limit, self.output_limit)

        self.prev_measured = measured
        return output

    def set_integral_limit(self, limit: float) -> None:
        self.i_limit = limit

    def reset(self, actual: float) -> None:
        self.error = 0.0
        self.prev_measured = actual
        self.integ = 0.0
        self.deriv = 0.0

    def set_desired(self, desired: float) -> None:
        self.desired = desired

    def is_active(self) -> bool:
        return not (self.kp < 0.0001 and self.ki < 0.0001 and self.kd < 0.0001)

    def set_dt(self, dt: float) -> None:
        self.dt = dt

    def reset_filter(self, sampling_rate: float, cutoff_freq: float, enable_d_filter: bool) -> None:
        self.enable_d_filter = enable_d_filter
        if self.enable_d_filter:
            self.d_filter.init(sampling_rate, cutoff_freq)
