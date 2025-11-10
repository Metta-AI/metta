"""Curriculum helpers for cycling through CoGames maps."""

from collections import deque
from typing import Callable

from mettagrid.config.mettagrid_config import MettaGridEnvConfig


class RotationSupplier:
    """Picklable callable that cycles through (name, cfg) tuples."""

    __slots__ = ("rotation",)

    def __init__(self, missions: list[tuple[str, MettaGridEnvConfig]]) -> None:
        self.rotation = deque(missions)

    def __call__(self) -> MettaGridEnvConfig:
        # Rotate left and return the new head in one step
        self.rotation.rotate(-1)
        return self.rotation[0][1]


def make_rotation(missions: list[tuple[str, MettaGridEnvConfig]]) -> Callable[[], MettaGridEnvConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")

    return RotationSupplier(missions)
