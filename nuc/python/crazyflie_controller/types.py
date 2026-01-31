from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class StabMode(IntEnum):
    DISABLE = 0
    ABS = 1
    VELOCITY = 2


@dataclass
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class Attitude:
    roll: float
    pitch: float
    yaw: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class AttitudeMode:
    x: StabMode = StabMode.DISABLE
    y: StabMode = StabMode.DISABLE
    z: StabMode = StabMode.DISABLE
    roll: StabMode = StabMode.DISABLE
    pitch: StabMode = StabMode.DISABLE
    yaw: StabMode = StabMode.DISABLE
    quat: StabMode = StabMode.DISABLE


@dataclass
class SensorData:
    gyro: Vec3
    acc: Vec3


@dataclass
class State:
    attitude: Attitude
    attitude_quat: Quaternion
    position: Vec3
    velocity: Vec3
    acc: Vec3


@dataclass
class Setpoint:
    attitude: Attitude
    attitude_rate: Attitude
    attitude_quat: Quaternion
    thrust: float
    position: Vec3
    velocity: Vec3
    acceleration: Vec3
    jerk: Vec3
    velocity_body: bool
    mode: AttitudeMode


@dataclass
class Control:
    roll: int
    pitch: int
    yaw: int
    thrust: float
    control_mode: str = "legacy"
