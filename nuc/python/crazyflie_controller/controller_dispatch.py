from __future__ import annotations

from enum import Enum

from .controller_pid import ControllerPID, RateContext
from .types import Control, SensorData, Setpoint, State


class ControllerType(Enum):
    AUTO = "auto"
    PID = "pid"
    MELLINGER = "mellinger"
    INDI = "indi"
    BRESCIANINI = "brescianini"
    LEE = "lee"


class ControllerDispatch:
    def __init__(self, controller_type: str, config: dict, rate_context: RateContext) -> None:
        self.controller_type = self._normalize(controller_type)
        self.pid = ControllerPID(config, rate_context)

    @staticmethod
    def _normalize(name: str) -> ControllerType:
        if not name:
            return ControllerType.AUTO
        name_l = name.lower()
        if name_l in ("pid", "controllerpid"):
            return ControllerType.PID
        if name_l in ("mellinger",):
            return ControllerType.MELLINGER
        if name_l in ("indi",):
            return ControllerType.INDI
        if name_l in ("brescianini", "bres"):
            return ControllerType.BRESCIANINI
        if name_l in ("lee",):
            return ControllerType.LEE
        return ControllerType.AUTO

    def update(self, control: Control, setpoint: Setpoint, sensors: SensorData, state: State, stabilizer_step: int) -> None:
        if self.controller_type in (ControllerType.AUTO, ControllerType.PID):
            self.pid.update(control, setpoint, sensors, state, stabilizer_step)
            return
        raise NotImplementedError(f"Controller type not implemented: {self.controller_type}")
