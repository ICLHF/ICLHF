from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class ActionType(Enum):
    OPEN = 0
    CLOSE = 1
    SLICE = 2
    THROW = 3
    CLEAN = 4
    FOLD = 5
    UNFOLD = 6
    HEAT = 7
    FREEZE = 8

    @classmethod
    def from_str(cls, s: str) -> ActionType:
        s = s.lower()
        for member in cls:
            if member.name.lower() == s:
                return member
        raise ValueError(f"{s!r} is not a valid {cls.__name__}")


class Env(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config: dict = config

    @property
    def config(self) -> dict:
        return self._config

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def play(self) -> None: ...

    @abstractmethod
    def current_time_step_index(self) -> int: ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def action(self, name: str, action_type: ActionType) -> None: ...

    @abstractmethod
    def get_bbox(self, name: str) -> list[float]: ...

    @abstractmethod
    def get_collisions(self, names: list[str]) -> list[tuple[str, str]]: ...

    @abstractmethod
    def set_view_pose(self, pose: tuple[list[float], list[float]]) -> None: ...

    @abstractmethod
    def get_position(self, name: str) -> list[float]: ...

    @abstractmethod
    def set_position(self, name: str, pos: list[float]) -> None: ...

    @abstractmethod
    def get_orientation(self, name: str) -> list[float]: ...

    @abstractmethod
    def set_orientation(self, name: str, ori: list[float]) -> None: ...

    def get_position_orientation(self, name: str) -> tuple[list[float], list[float]]:
        return (self.get_position(name), self.get_orientation(name))

    def set_position_orientation(
        self, name: str, pose: tuple[list[float], list[float]]
    ):
        self.set_position(name, pose[0])
        self.set_orientation(name, pose[1])
