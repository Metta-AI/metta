from collections.abc import Mapping, Sequence
from typing import Iterable

from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    Position,
    RecipeConfig,
)

resources = [
    "energy",
    "carbon",
    "oxygen",
    "geranium",
    "silicon",
    "heart",
    "disruptor",
    "modulator",
    "resonator",
    "scrabbler",
]


def _recipe(
    *,
    positions: Iterable[Position] = ("Any",),
    inputs: Mapping[str, int] | None = None,
    outputs: Mapping[str, int] | None = None,
    cooldown: int = 1,
) -> tuple[list[Position], RecipeConfig]:
    """Create a recipe tuple while normalising common defaults."""

    return (
        list(positions),
        RecipeConfig(
            input_resources=dict(inputs or {}),
            output_resources=dict(outputs or {}),
            cooldown=cooldown,
        ),
    )


def make_assembler(
    *,
    name: str,
    type_id: int,
    recipes: Sequence[tuple[Iterable[Position], Mapping[str, int], Mapping[str, int], int]] | None = None,
    recipe_defs: Sequence[tuple[list[Position], RecipeConfig]] | None = None,
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    """Create an assembler with optional pre-built recipe definitions.

    The `recipes` convenience argument accepts tuples of (positions, inputs, outputs, cooldown) to
    reduce boilerplate when defining lots of stations inside `scenarios.py`.
    """

    if recipes and recipe_defs:
        raise ValueError("Provide either `recipes` or `recipe_defs`, not both.")

    resolved_recipes: list[tuple[list[Position], RecipeConfig]]
    if recipe_defs is not None:
        resolved_recipes = [(_positions, _recipe_cfg) for _positions, _recipe_cfg in recipe_defs]
    elif recipes is not None:
        resolved_recipes = [
            _recipe(positions=positions, inputs=inputs, outputs=outputs, cooldown=cooldown)
            for positions, inputs, outputs, cooldown in recipes
        ]
    else:
        resolved_recipes = []

    return AssemblerConfig(
        name=name,
        type_id=type_id,
        recipes=resolved_recipes,
        tags=list(tags or []),
    )


def make_resource_extractor(
    *,
    resource_name: str,
    energy_cost: int,
    output_per_cycle: int,
    cooldown: int,
    type_id: int,
    name: str,
    positions: Iterable[Position] = ("Any",),
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    """Factory for resource extractors with custom production values."""

    return make_assembler(
        name=name,
        type_id=type_id,
        recipes=[
            (
                list(positions),
                {"energy": energy_cost} if energy_cost else {},
                {resource_name: output_per_cycle},
                cooldown,
            )
        ],
        tags=tags,
    )


def make_charger(
    *,
    energy_output: int = 50,
    cooldown: int = 1,
    positions: Iterable[Position] = ("Any",),
    type_id: int = 5,
    name: str = "charger",
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    """Create a charger that injects energy into visiting agents."""

    return make_assembler(
        name=name,
        type_id=type_id,
        recipes=[(list(positions), {}, {"energy": energy_output}, cooldown)],
        tags=tags,
    )


def make_carbon_extractor(
    *,
    energy_cost: int = 1,
    output_per_cycle: int = 1,
    cooldown: int = 1,
    type_id: int = 2,
    name: str = "carbon_extractor",
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    return make_resource_extractor(
        resource_name="carbon",
        energy_cost=energy_cost,
        output_per_cycle=output_per_cycle,
        cooldown=cooldown,
        type_id=type_id,
        name=name,
        tags=tags,
    )


def make_oxygen_extractor(
    *,
    energy_cost: int = 1,
    output_per_cycle: int = 10,
    cooldown: int = 1,
    type_id: int = 3,
    name: str = "oxygen_extractor",
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    return make_resource_extractor(
        resource_name="oxygen",
        energy_cost=energy_cost,
        output_per_cycle=output_per_cycle,
        cooldown=cooldown,
        type_id=type_id,
        name=name,
        tags=tags,
    )


def make_geranium_extractor(
    *,
    energy_cost: int = 1,
    output_per_cycle: int = 10,
    cooldown: int = 100,
    type_id: int = 4,
    name: str = "geranium_extractor",
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    # TODO: Expose depletion curves once engine supports per-node decay hooks.
    return make_resource_extractor(
        resource_name="geranium",
        energy_cost=energy_cost,
        output_per_cycle=output_per_cycle,
        cooldown=cooldown,
        type_id=type_id,
        name=name,
        tags=tags,
    )


def make_silicon_extractor(
    *,
    energy_cost: int = 10,
    output_per_cycle: int = 1,
    cooldown: int = 1,
    type_id: int = 15,
    name: str = "silicon_extractor",
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    return make_resource_extractor(
        resource_name="silicon",
        energy_cost=energy_cost,
        output_per_cycle=output_per_cycle,
        cooldown=cooldown,
        type_id=type_id,
        name=name,
        tags=tags,
    )


def make_chest(
    *,
    resource_type: str = "heart",
    deposit_positions: Sequence[Position] = ("E",),
    withdrawal_positions: Sequence[Position] = ("W",),
    type_id: int = 17,
    tags: Sequence[str] | None = None,
) -> ChestConfig:
    return ChestConfig(
        type_id=type_id,
        resource_type=resource_type,
        deposit_positions=list(deposit_positions),
        withdrawal_positions=list(withdrawal_positions),
        tags=list(tags or []),
    )


def make_core_assembler(
    *,
    energy_cost: int = 3,
    heart_output: int = 1,
    cooldown: int = 1,
    type_id: int = 8,
    name: str = "assembler",
    positions: Iterable[Position] = ("Any",),
    tags: Sequence[str] | None = None,
) -> AssemblerConfig:
    return make_assembler(
        name=name,
        type_id=type_id,
        recipes=[
            (
                list(positions),
                {"energy": energy_cost},
                {"heart": heart_output},
                cooldown,
            )
        ],
        tags=tags,
    )


# Backwards-compatible helpers used in early prototypes ---------------------


def charger() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_charger()


def carbon_extractor() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_carbon_extractor()


def oxygen_extractor() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_oxygen_extractor()


def geranium_extractor() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_geranium_extractor()


def silicon_extractor() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_silicon_extractor()


def chest() -> ChestConfig:  # pragma: no cover - legacy shim
    return make_chest()


def assembler() -> AssemblerConfig:  # pragma: no cover - legacy shim
    return make_core_assembler()
