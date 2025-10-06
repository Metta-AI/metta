"""Curriculum helpers for cycling through CoGames maps."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from cogames import game
from cogames.cogs_vs_clips import scenarios
from mettagrid.config.mettagrid_config import MettaGridConfig

_BASE_ROTATION: tuple[str, ...] = (
    "training_facility_1",
    "training_facility_2",
    "training_facility_3",
    "training_facility_4",
    "training_facility_5",
    "training_facility_6",
    "machina_1",
    "machina_2",
)


def _make_rotation_supplier(
    names: Iterable[str] | None,
    *,
    easy: bool = False,
    shaped: bool = False,
) -> Callable[[], MettaGridConfig]:
    rotation = deque(tuple(names) if names is not None else _BASE_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        cfg = game.get_game(map_name).model_copy(deep=True)
        if easy:
            scenarios.add_easy_heart_recipe(cfg)
        if shaped:
            scenarios.add_shaped_rewards(cfg)
        if easy or shaped:
            scenarios.extend_max_steps(cfg)
        return cfg

    return _supplier


def training_rotation(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Cycle the default training rotation."""

    return _make_rotation_supplier(names, easy=False, shaped=False)


def training_rotation_easy(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Cycle the rotation with easy heart crafting enabled."""

    return _make_rotation_supplier(names, easy=True, shaped=False)


def training_rotation_shaped(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Cycle the rotation with shaped intermediate rewards."""

    return _make_rotation_supplier(names, easy=False, shaped=True)


def training_rotation_easy_shaped(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Cycle the rotation with both easy hearts and shaped rewards."""

    return _make_rotation_supplier(names, easy=True, shaped=True)


# Backward compatibility alias
training_facility_rotation = training_rotation
