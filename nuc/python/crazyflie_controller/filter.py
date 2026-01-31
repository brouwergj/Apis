from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Lpf2pData:
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    delay_1: float = 0.0
    delay_2: float = 0.0

    def init(self, sample_freq: float, cutoff_freq: float) -> None:
        if cutoff_freq <= 0.0:
            return
        self.set_cutoff(sample_freq, cutoff_freq)

    def set_cutoff(self, sample_freq: float, cutoff_freq: float) -> None:
        if cutoff_freq <= 0.0:
            return
        fr = sample_freq / cutoff_freq
        ohm = math.tan(math.pi / fr)
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm
        self.b0 = (ohm * ohm) / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c
        self.delay_1 = 0.0
        self.delay_2 = 0.0

    def apply(self, sample: float) -> float:
        delay_0 = sample - self.delay_1 * self.a1 - self.delay_2 * self.a2
        if not math.isfinite(delay_0):
            delay_0 = sample
        output = delay_0 * self.b0 + self.delay_1 * self.b1 + self.delay_2 * self.b2
        self.delay_2 = self.delay_1
        self.delay_1 = delay_0
        return output

    def reset(self, sample: float) -> float:
        denom = self.b0 + self.b1 + self.b2
        if denom == 0.0:
            self.delay_1 = sample
            self.delay_2 = sample
            return sample
        dval = sample / denom
        self.delay_1 = dval
        self.delay_2 = dval
        return self.apply(sample)
