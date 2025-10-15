"""Curriculum helpers for cycling through CoGames maps."""

from collections import deque
from typing import Callable

from mettagrid.config.mettagrid_config import MettaGridConfig


def make_rotation(missions: list[tuple[str, MettaGridConfig]]) -> Callable[[], MettaGridConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")
    rotation = deque(missions)

    def supplier() -> MettaGridConfig:
        _, cfg = rotation[0]
        rotation.rotate(-1)
        return cfg

    return supplier
