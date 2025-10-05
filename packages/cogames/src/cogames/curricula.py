"""Curriculum helpers for cycling through CoGames maps."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from cogames import game
from mettagrid.config.mettagrid_config import MettaGridConfig, RecipeConfig

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

    def _ensure_simple_heart_recipe(cfg: MettaGridConfig) -> None:
        assembler_cfg = cfg.game.objects.get("assembler")
        if assembler_cfg is None:
            return

        # Check whether an energy-only heart recipe already exists
        for _, recipe in assembler_cfg.recipes:
            if recipe.output_resources.get("heart"):
                inputs = recipe.input_resources or {}
                if set(inputs.keys()) == {"energy"} and inputs.get("energy", 0) <= 1:
                    return
                break

        heart_recipe = RecipeConfig(
            input_resources={"energy": 1},
            output_resources={"heart": 1},
            cooldown=1,
        )
        assembler_cfg.recipes.insert(0, (["Any"], heart_recipe))

    def _supplier() -> MettaGridConfig:
        map_name = rotation[0]
        rotation.rotate(-1)
        cfg = game.get_game(map_name).model_copy(deep=True)
        _ensure_simple_heart_recipe(cfg)
        return cfg

    return _supplier
