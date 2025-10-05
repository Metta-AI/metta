"""Curriculum helpers for cycling through CoGames maps."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from cogames import game
from mettagrid.config.mettagrid_config import MettaGridConfig

_DEFAULT_ROTATION: tuple[str, ...] = (
    "training_facility_1",
    "training_facility_2",
    "training_facility_3",
    "training_facility_4",
    "training_facility_5",
    "training_facility_6",
    "machina_1",
    "machina_2",
)

def training_rotation(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Create a supplier that cycles the default training rotation."""

    rotation = deque(tuple(names) if names is not None else _DEFAULT_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier
