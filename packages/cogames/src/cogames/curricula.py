"""Curriculum helpers for cycling through CoGames maps."""

from collections import deque
from typing import Callable

from mettagrid.config.mettagrid_config import MettaGridConfig


class RotationSupplier:
    """Picklable callable that cycles through (name, cfg) tuples."""

    __slots__ = ("rotation",)

    def __init__(self, missions: list[tuple[str, MettaGridConfig]]) -> None:
        self.rotation = deque(missions)

    def __call__(self) -> MettaGridConfig:
        _, cfg = self.rotation[0]
        self.rotation.rotate(-1)
        return cfg


def make_rotation(missions: list[tuple[str, MettaGridConfig]]) -> Callable[[], MettaGridConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")

    return RotationSupplier(missions)
