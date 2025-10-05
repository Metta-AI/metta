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

_EASY_ROTATION: tuple[str, ...] = (
    "training_facility_1_easy",
    "training_facility_2_easy",
    "training_facility_3_easy",
    "training_facility_4_easy",
    "training_facility_5_easy",
    "training_facility_6_easy",
    "machina_1_easy",
    "machina_2_easy",
)

_SHAPED_ROTATION: tuple[str, ...] = (
    "training_facility_1_shaped",
    "training_facility_2_shaped",
    "training_facility_3_shaped",
    "training_facility_4_shaped",
    "training_facility_5_shaped",
    "training_facility_6_shaped",
    "machina_1_shaped",
    "machina_2_shaped",
)

_EASY_SHAPED_ROTATION: tuple[str, ...] = (
    "training_facility_1_easy_shaped",
    "training_facility_2_easy_shaped",
    "training_facility_3_easy_shaped",
    "training_facility_4_easy_shaped",
    "training_facility_5_easy_shaped",
    "training_facility_6_easy_shaped",
    "machina_1_easy_shaped",
    "machina_2_easy_shaped",
)

_EASY_ROTATION: tuple[str, ...] = (
    "training_facility_1_easy",
    "training_facility_2_easy",
    "training_facility_3_easy",
    "training_facility_4_easy",
    "training_facility_5_easy",
    "training_facility_6_easy",
    "machina_1_easy",
    "machina_2_easy",
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


def training_rotation_easy(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Create a supplier that cycles the easy-heart training rotation."""

    rotation = deque(tuple(names) if names is not None else _EASY_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier


def training_rotation_shaped(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Create a supplier that cycles the shaped-reward training rotation."""

    rotation = deque(tuple(names) if names is not None else _SHAPED_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier


def training_rotation_easy_shaped(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Create a supplier that cycles the easy-heart + shaped training rotation."""

    rotation = deque(tuple(names) if names is not None else _EASY_SHAPED_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier


def training_rotation_easy(names: Iterable[str] | None = None) -> Callable[[], MettaGridConfig]:
    """Create a supplier that cycles the easy-heart training rotation."""

    rotation = deque(tuple(names) if names is not None else _EASY_ROTATION)
    if not rotation:
        raise ValueError("Rotation must contain at least one game name")

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        return game.get_game(map_name).model_copy(deep=True)

    return _supplier
