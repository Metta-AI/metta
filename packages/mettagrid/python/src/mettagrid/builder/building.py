from typing import Literal

from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, ConverterConfig, RecipeConfig, WallConfig

wall = WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛")
block = WallConfig(name="block", type_id=14, map_char="s", render_symbol="📦", swappable=True)

altar = ConverterConfig(
    name="altar",
    type_id=8,
    map_char="_",
    render_symbol="🎯",
    input_resources={"battery_red": 3},
    output_resources={"heart": 1},
    cooldown=10,
)


def make_mine(color: str, type_id: int) -> ConverterConfig:
    char_map = {"red": "m", "blue": "b", "green": "g"}
    symbol_map = {"red": "🔺", "blue": "🔷", "green": "💚"}
    return ConverterConfig(
        name=f"mine_{color}",
        type_id=type_id,
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        output_resources={f"ore_{color}": 1},
        cooldown=50,
    )


mine_red = make_mine("red", 2)
mine_blue = make_mine("blue", 3)
mine_green = make_mine("green", 4)


def make_generator(color: str, type_id: int) -> ConverterConfig:
    char_map = {"red": "n", "blue": "B", "green": "G"}
    symbol_map = {"red": "🔋", "blue": "🔌", "green": "🟢"}
    return ConverterConfig(
        name=f"generator_{color}",
        type_id=type_id,
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        input_resources={f"ore_{color}": 1},
        output_resources={f"battery_{color}": 1},
        cooldown=25,
    )


generator_red = make_generator("red", 5)
generator_blue = make_generator("blue", 6)
generator_green = make_generator("green", 7)

lasery = ConverterConfig(
    name="lasery",
    type_id=15,
    map_char="S",
    render_symbol="🟥",
    input_resources={"battery_red": 1, "ore_red": 2},
    output_resources={"laser": 1},
    cooldown=10,
)

armory = ConverterConfig(
    name="armory",
    type_id=16,
    map_char="o",
    render_symbol="🔵",
    input_resources={"ore_red": 3},
    output_resources={"armor": 1},
    cooldown=10,
)

# Assembler building definitions
assembler_altar = AssemblerConfig(
    name="altar",
    type_id=8,
    map_char="_",
    render_symbol="🎯",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"battery_red": 3},
                output_resources={"heart": 1},
                cooldown=10,
            ),
        )
    ],
)


def make_assembler_mine(color: str, type_id: int) -> AssemblerConfig:
    char_map = {"red": "m", "blue": "b", "green": "g"}
    symbol_map = {"red": "🔺", "blue": "🔷", "green": "💚"}
    return AssemblerConfig(
        name=f"mine_{color}",
        type_id=type_id,
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={f"ore_{color}": 1},
                    cooldown=50,
                ),
            )
        ],
    )


assembler_mine_red = make_assembler_mine("red", 2)
assembler_mine_blue = make_assembler_mine("blue", 3)
assembler_mine_green = make_assembler_mine("green", 4)


def make_assembler_generator(color: str, type_id: int) -> AssemblerConfig:
    char_map = {"red": "n", "blue": "B", "green": "G"}
    symbol_map = {"red": "🔋", "blue": "🔌", "green": "🟢"}
    return AssemblerConfig(
        name=f"generator_{color}",
        type_id=type_id,
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={f"ore_{color}": 1},
                    output_resources={f"battery_{color}": 1},
                    cooldown=25,
                ),
            )
        ],
    )


assembler_generator_red = make_assembler_generator("red", 5)
assembler_generator_blue = make_assembler_generator("blue", 6)
assembler_generator_green = make_assembler_generator("green", 7)

assembler_lasery = AssemblerConfig(
    name="lasery",
    type_id=15,
    map_char="S",
    render_symbol="🟥",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"battery_red": 1, "ore_red": 2},
                output_resources={"laser": 1},
                cooldown=10,
            ),
        )
    ],
)

assembler_armory = AssemblerConfig(
    name="armory",
    type_id=16,
    map_char="o",
    render_symbol="🔵",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"ore_red": 3},
                output_resources={"armor": 1},
                cooldown=10,
            ),
        )
    ],
)

assembler_lab = AssemblerConfig(
    name="lab",
    type_id=17,
    map_char="L",
    render_symbol="🔵",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"battery_red": 1, "ore_red": 2},
                output_resources={"laser": 1},
                cooldown=10,
            ),
        )
    ],
)

assembler_factory = AssemblerConfig(
    name="factory",
    type_id=18,
    map_char="F",
    render_symbol="🟪",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"battery_red": 1, "ore_red": 2},
                output_resources={"laser": 1},
                cooldown=10,
            ),
        )
    ],
)

assembler_temple = AssemblerConfig(
    name="temple",
    type_id=19,
    map_char="T",
    render_symbol="🟨",
    recipes=[
        (
            ["Any"],
            RecipeConfig(
                input_resources={"battery_red": 1, "ore_red": 2},
                output_resources={"laser": 1},
                cooldown=10,
            ),
        )
    ],
)


# Chest building definitions. Maybe not needed beyond the raw config?
def make_chest(
    resource_type: str,
    type_id: int,
    name: str = "chest",
    map_char: str = "C",
    render_symbol: str = "📦",
    position_deltas: list[tuple[Literal["NW", "N", "NE", "W", "E", "SW", "S", "SE"], int]] | None = None,
    initial_inventory: int = 0,
    max_inventory: int = 255,
) -> ChestConfig:
    """Create a chest configuration for a specific resource type.

    Args:
        resource_type: Resource type that this chest can store
        type_id: Unique type ID
        name: Name of the chest
        map_char: Character for ASCII maps
        render_symbol: Symbol for rendering
        position_deltas: List of (position, delta) tuples. Positive delta = deposit amount, negative = withdraw amount
        initial_inventory: Initial amount of resource_type in the chest
        max_inventory: Maximum inventory (255 = default, -1 = unlimited, resources destroyed when full)
    """
    if position_deltas is None:
        position_deltas = []  # Default to no positions configured

    return ChestConfig(
        name=name,
        type_id=type_id,
        map_char=map_char,
        render_symbol=render_symbol,
        resource_type=resource_type,
        position_deltas=position_deltas,
        initial_inventory=initial_inventory,
        max_inventory=max_inventory,
    )


# Example chest configurations
chest_heart = make_chest("heart", 20, position_deltas=[("N", 1), ("S", -1)])
