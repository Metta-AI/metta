"""Runtime helpers for validating MettaGrid gymnasium spaces."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
from gymnasium import spaces

SpaceT = TypeVar("SpaceT", bound=spaces.Space)


def _require_space(space: spaces.Space, kind: type[SpaceT], label: str) -> SpaceT:
    if not isinstance(space, kind):
        raise TypeError(
            f"MettaGrid {label} space must be {kind.__name__}, got {type(space).__name__}",
        )
    return cast(SpaceT, space)


def validate_observation_space(space: spaces.Space) -> None:
    box = _require_space(space, spaces.Box, "observation")
    if box.dtype != np.uint8:
        raise TypeError(
            f"MettaGrid observation space must have dtype uint8, got {box.dtype}",
        )


def validate_action_space(space: spaces.Space) -> None:
    _require_space(space, spaces.Discrete, "action")


def get_observation_shape(space: spaces.Box) -> tuple[int, ...]:
    return tuple(space.shape)


def get_action_count(space: spaces.Discrete) -> int:
    return int(space.n)
