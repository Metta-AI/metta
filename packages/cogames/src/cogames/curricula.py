"""Curriculum helpers for cycling through CoGames maps."""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable

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

_DEFAULT_SEQUENCE: Deque[str] = deque(_DEFAULT_ROTATION)
_DEFAULT_AGENT_COUNT = game.get_game(_DEFAULT_SEQUENCE[0]).game.num_agents


def alternating(sequence: Deque[str] | None = None) -> MettaGridConfig:
    """Return the next map in the rotation, excluding large dungeon boards."""

    queue = sequence.copy() if sequence is not None else _DEFAULT_SEQUENCE.copy()
    if not queue:
        raise ValueError("Curriculum sequence must contain at least one game name")

    for _ in range(len(queue)):
        map_name = queue[0]
        queue.rotate(-1)
        cfg = game.get_game(map_name)
        if cfg.game.num_agents == _DEFAULT_AGENT_COUNT:
            return cfg

    raise ValueError("Curriculum contains no maps with the expected agent count")


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
