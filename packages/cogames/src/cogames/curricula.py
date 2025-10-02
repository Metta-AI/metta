"""Curriculum helpers for cycling through small CoGames boards."""

from collections import deque
from typing import Deque, Iterable

from cogames import game
from mettagrid.config.mettagrid_config import MettaGridConfig

_DEFAULT_SEQUENCE: Deque[str] = deque(
    [
        "assembler_2_simple",
        "assembler_2_complex",
        "machina_1",
    ]
)
_DEFAULT_AGENT_COUNT = game.get_game(_DEFAULT_SEQUENCE[0]).game.num_agents

_TRAINING_ROTATION_NAMES = (
    "training_facility_1",
    "training_facility_2",
    "training_facility_3",
    "training_facility_4",
    "machina_1",
)


def alternating(sequence: Deque[str] | None = None) -> MettaGridConfig:
    """Return the next map in the rotation, excluding large dungeon boards."""

    queue = sequence if sequence is not None else _DEFAULT_SEQUENCE
    if not queue:
        raise ValueError("Curriculum sequence must contain at least one game name")

    for _ in range(len(queue)):
        map_name = queue[0]
        queue.rotate(-1)
        cfg = game.get_game(map_name)
        if cfg.game.num_agents == _DEFAULT_AGENT_COUNT:
            return cfg

    raise ValueError("Curriculum contains no maps with the expected agent count")


def training_facility_rotation(names: Iterable[str] | None = None):
    """Create a supplier that cycles training facility boards and machina_1."""

    rotation = deque(tuple(names) if names is not None else _TRAINING_ROTATION_NAMES)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier
