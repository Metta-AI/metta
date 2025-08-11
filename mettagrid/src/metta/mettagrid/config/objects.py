"""Predefined MettaGrid objects for demos.

This module provides common MettaGrid objects with sensible defaults,
similar to the old config/object.py approach but using the new Pydantic models.
"""

from metta.mettagrid.config.mettagrid_config import PyConverterConfig, PyWallConfig

# Basic objects
wall = PyWallConfig(type_id=1, swappable=False)
block = PyWallConfig(type_id=14, swappable=True)

# Altar - converts batteries to hearts
altar = PyConverterConfig(
    type_id=8,
    input_resources={"battery_red": 2},
    output_resources={"heart": 1},
    max_output=5,
    conversion_ticks=1,
    cooldown=20,
    initial_resource_count=1,
    color=2,
)


def make_mine(color: str, type_id: int) -> PyConverterConfig:
    """Create a mine that produces ore of the specified color."""
    return PyConverterConfig(
        type_id=type_id,
        output_resources={f"ore_{color}": 1},
        max_output=-1,
        conversion_ticks=1,
        cooldown=3,
        initial_resource_count=0,
        color=0,
    )


def make_generator(color: str, type_id: int) -> PyConverterConfig:
    """Create a generator that converts ore to batteries of the specified color."""
    return PyConverterConfig(
        type_id=type_id,
        input_resources={f"ore_{color}": 1},
        output_resources={f"battery_{color}": 1},
        max_output=-1,
        conversion_ticks=1,
        cooldown=2,
        initial_resource_count=0,
        color=0,
    )


# Mines
mine_red = make_mine("red", 2)
mine_blue = make_mine("blue", 3)
mine_green = make_mine("green", 4)

# Generators
generator_red = make_generator("red", 5)
generator_blue = make_generator("blue", 6)
generator_green = make_generator("green", 7)

# Combat objects
lasery = PyConverterConfig(
    type_id=15,
    input_resources={"battery_red": 1, "ore_red": 2},
    output_resources={"laser": 1},
    max_output=-1,
    conversion_ticks=1,
    cooldown=10,
    initial_resource_count=0,
    color=0,
)

armory = PyConverterConfig(
    type_id=16,
    input_resources={"ore_red": 3},
    output_resources={"armor": 1},
    max_output=-1,
    conversion_ticks=1,
    cooldown=10,
    initial_resource_count=0,
    color=0,
)
