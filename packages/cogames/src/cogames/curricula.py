"""Curriculum helpers for cycling through CoGames maps."""

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Tuple

from mettagrid.config.mettagrid_config import MettaGridConfig


def make_rotation(missions: list[tuple[str, MettaGridConfig]]) -> Callable[[], MettaGridConfig]:
    if not missions:
        raise ValueError("Must have at least one mission in rotation")

    # Use a picklable callable to support multiprocessing (spawn) backends.
    @dataclass
    class _RotationSupplier:
        rotation: deque  # deque of (name, cfg) tuples

        def __call__(self) -> MettaGridConfig:
            _, cfg = self.rotation[0]
            self.rotation.rotate(-1)
            return cfg

    return _RotationSupplier(deque(missions))
