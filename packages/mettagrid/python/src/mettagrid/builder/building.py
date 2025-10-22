from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    FixedPosition,
    ProtocolConfig,
    WallConfig,
)

wall = WallConfig(name="wall", map_char="#", render_symbol="⬛")
block = WallConfig(name="block", map_char="s", render_symbol="📦", swappable=True)

# Assembler building definitions
assembler_altar = AssemblerConfig(
    name="altar",
    map_char="_",
    render_symbol="🎯",
    recipes=[([], ProtocolConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))],
)


def make_assembler_mine(color: str) -> AssemblerConfig:
    char_map = {"red": "m", "blue": "b", "green": "g"}
    symbol_map = {"red": "🔺", "blue": "🔷", "green": "💚"}
    return AssemblerConfig(
        name=f"mine_{color}",
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        recipes=[([], ProtocolConfig(output_resources={f"ore_{color}": 1}, cooldown=50))],
    )


assembler_mine_red = make_assembler_mine("red")
assembler_mine_blue = make_assembler_mine("blue")
assembler_mine_green = make_assembler_mine("green")


def make_assembler_generator(color: str) -> AssemblerConfig:
    char_map = {"red": "n", "blue": "B", "green": "G"}
    symbol_map = {"red": "🔋", "blue": "🔌", "green": "🟢"}
    return AssemblerConfig(
        name=f"generator_{color}",
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        recipes=[
            (
                [],
                ProtocolConfig(
                    input_resources={f"ore_{color}": 1}, output_resources={f"battery_{color}": 1}, cooldown=25
                ),
            )
        ],
    )


assembler_generator_red = make_assembler_generator("red")
assembler_generator_blue = make_assembler_generator("blue")
assembler_generator_green = make_assembler_generator("green")

assembler_lasery = AssemblerConfig(
    name="lasery",
    map_char="S",
    render_symbol="🟥",
    recipes=[
        (
            [],
            ProtocolConfig(
                input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10
            ),
        )
    ],
)

assembler_armory = AssemblerConfig(
    name="armory",
    map_char="o",
    render_symbol="🔵",
    recipes=[([], ProtocolConfig(input_resources={"ore_red": 3}, output_resources={"armor": 1}, cooldown=10))],
)

assembler_lab = AssemblerConfig(
    name="lab",
    map_char="L",
    render_symbol="🔵",
    recipes=[
        (
            [],
            ProtocolConfig(
                input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10
            ),
        )
    ],
)

assembler_factory = AssemblerConfig(
    name="factory",
    map_char="F",
    render_symbol="🟪",
    recipes=[
        (
            [],
            ProtocolConfig(
                input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10
            ),
        )
    ],
)

assembler_temple = AssemblerConfig(
    name="temple",
    map_char="T",
    render_symbol="🟨",
    recipes=[
        (
            [],
            ProtocolConfig(
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
    initial_inventory: int = 0,
    position_deltas: list[tuple[FixedPosition, int]] | None = None,
    max_inventory: int = 255,
    name: str | None = None,
    map_char: str = "C",
    render_symbol: str = "📦",
) -> ChestConfig:
    """Create a chest configuration for a specific resource type.

    Args:
        resource_type: Resource type that this chest can store
        name: Name of the chest
        map_char: Character for ASCII maps
        render_symbol: Symbol for rendering
        position_deltas: List of (position, delta) tuples. Positive delta = deposit amount, negative = withdraw amount
        initial_inventory: Initial amount of resource_type in the chest
        max_inventory: Maximum inventory (255 = default, -1 = unlimited, resources destroyed when full)
    """
    if position_deltas is None:
        position_deltas = []

    if name is None:
        name = f"chest_{resource_type}"

    return ChestConfig(
        name=name,
        map_char=map_char,
        render_symbol=render_symbol,
        resource_type=resource_type,
        position_deltas=position_deltas,
        initial_inventory=initial_inventory,
        max_inventory=max_inventory,
    )


# Example chest configurations
chest_heart = make_chest("heart", position_deltas=[("N", 1), ("S", -1)])

nav_assembler = AssemblerConfig(
    name="nav_assembler",
    map_char="_",
    render_symbol="🛣️",
    recipes=[([], ProtocolConfig(input_resources={}, output_resources={"heart": 1}, cooldown=255))],
)
