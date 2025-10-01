"""Minimal curricula helpers for alternating between predefined maps."""

from collections import deque
from typing import Deque

from cogames import game
from mettagrid.config.mettagrid_config import MettaGridConfig

_DEFAULT_SEQUENCE: Deque[str] = deque(
    [
        "assembler_1_simple",
        "assembler_2_simple",
        "machina_1",
    ]
)


def alternating(sequence: Deque[str] | None = None) -> MettaGridConfig:
    """Return the next MettaGridConfig, cycling through the supplied sequence.

    When ``sequence`` is omitted, the default map rotation is used.
    """

    queue = sequence if sequence is not None else _DEFAULT_SEQUENCE
    if not queue:
        raise ValueError("Curriculum sequence must contain at least one game name")

    next_map = queue[0]
    queue.rotate(-1)
    return game.get_game(next_map)
