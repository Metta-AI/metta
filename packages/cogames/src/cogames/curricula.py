"""Curricula helpers for rotating through small CoGames boards."""

from collections import deque
from typing import Deque, Iterable

from cogames import game
from mettagrid.config.mettagrid_config import MettaGridConfig

# We skip the "_big" / "_bigger" dungeon maps by default so training stays lightweight.
_DEFAULT_MAPS = (
    "assembler_2_simple",
    "assembler_2_complex",
    "machina_1",
)
_DEFAULT_SEQUENCE: Deque[str] = deque(_DEFAULT_MAPS)
_DEFAULT_AGENT_COUNT = game.get_game(_DEFAULT_SEQUENCE[0]).game.num_agents


def alternating(
    sequence: Iterable[str] | None = None,
    expected_agents: int | None = None,
) -> MettaGridConfig:
    """Return the next map, cycling through the supplied sequence.

    Parameters
    ----------
    sequence:
        Optional iterable of map names. If omitted, we use a small rotation that excludes
        the large ``machina_*_big`` boards. The iterable is materialised into a deque the
        first time this function runs so passes can be repeated safely.
    expected_agents:
        If provided, only maps whose ``game.num_agents`` matches this count are returned.
        By default we use the agent count of the first map in the sequence.
    """

    if sequence is None:
        queue: Deque[str] = _DEFAULT_SEQUENCE
    else:
        queue = deque(sequence) if not isinstance(sequence, deque) else sequence

    if not queue:
        raise ValueError("Curriculum sequence must contain at least one game name")

    target_agents = expected_agents if expected_agents is not None else _DEFAULT_AGENT_COUNT

    for _ in range(len(queue)):
        map_name = queue[0]
        queue.rotate(-1)
        cfg = game.get_game(map_name)
        if cfg.game.num_agents == target_agents:
            return cfg

    raise ValueError(
        "Curriculum does not contain any maps with the expected agent count "
        f"({target_agents})."
    )
