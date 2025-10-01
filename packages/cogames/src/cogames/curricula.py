"""Minimal curricula helpers for alternating between predefined maps."""

from collections import deque
from typing import Deque

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


def alternating(
    sequence: Deque[str] | None = None,
    expected_agents: int | None = None,
) -> MettaGridConfig:
    """Return the next MettaGridConfig, cycling through the supplied sequence.

    All returned configs share the same ``game.num_agents``. If ``expected_agents``
    is provided, only maps matching that count will be emitted.
    """

    queue = sequence if sequence is not None else _DEFAULT_SEQUENCE
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
