from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateSupervisor:
    count: int = 0
    expected_min: int = 0
    expected_max: int = 0
    next_eval_time_ms: int = 0
    eval_interval_ms: int = 0
    latest_count: int = 0
    skip: int = 0

    def init(self, os_time_ms: int, evaluation_interval_ms: int, min_count: int, max_count: int, skip: int) -> None:
        self.count = 0
        self.expected_min = min_count
        self.expected_max = max_count
        self.next_eval_time_ms = os_time_ms + evaluation_interval_ms
        self.eval_interval_ms = evaluation_interval_ms
        self.latest_count = 0
        self.skip = skip

    def validate(self, os_time_ms: int) -> bool:
        self.count += 1
        if os_time_ms < self.next_eval_time_ms:
            return True

        self.latest_count = self.count
        self.count = 0
        self.next_eval_time_ms += self.eval_interval_ms

        if self.skip > 0:
            self.skip -= 1
            return True

        return self.expected_min <= self.latest_count <= self.expected_max

    def latest(self) -> int:
        return self.latest_count
