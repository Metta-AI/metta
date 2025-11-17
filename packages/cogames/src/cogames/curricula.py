"""Curriculum helpers for cycling through CoGames maps."""

from collections import deque
from dataclasses import dataclass
from typing import Callable

from mettagrid.config.mettagrid_config import MettaGridConfig


@dataclass
class RotationSupplier:
    """Picklable callable that cycles through (name, cfg) tuples."""

    rotation: deque

    def __call__(self) -> MettaGridConfig:
        _, cfg = self.rotation[0]
        self.rotation.rotate(-1)
        return cfg


def make_rotation(missions: list[tuple[str, MettaGridConfig]]) -> Callable[[], MettaGridConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")

    return RotationSupplier(deque(missions))
