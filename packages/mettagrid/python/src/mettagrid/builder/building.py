from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
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
    protocols=[ProtocolConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10)],
)


def make_assembler_mine(color: str) -> AssemblerConfig:
    char_map = {"red": "m", "blue": "b", "green": "g"}
    symbol_map = {"red": "🔺", "blue": "🔷", "green": "💚"}
    return AssemblerConfig(
        name=f"mine_{color}",
        map_char=char_map[color],
        render_symbol=symbol_map[color],
        protocols=[ProtocolConfig(output_resources={f"ore_{color}": 1}, cooldown=50)],
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
        protocols=[
            ProtocolConfig(input_resources={f"ore_{color}": 1}, output_resources={f"battery_{color}": 1}, cooldown=25)
        ],
    )


assembler_generator_red = make_assembler_generator("red")
assembler_generator_blue = make_assembler_generator("blue")
assembler_generator_green = make_assembler_generator("green")

assembler_lasery = AssemblerConfig(
    name="lasery",
    map_char="S",
    render_symbol="🟥",
    protocols=[
        ProtocolConfig(input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10)
    ],
)

assembler_armory = AssemblerConfig(
    name="armory",
    map_char="o",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={"ore_red": 3}, output_resources={"armor": 1}, cooldown=10)],
)

assembler_lab = AssemblerConfig(
    name="lab",
    map_char="L",
    render_symbol="🔵",
    protocols=[
        ProtocolConfig(input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10)
    ],
)

assembler_factory = AssemblerConfig(
    name="factory",
    map_char="F",
    render_symbol="🟪",
    protocols=[
        ProtocolConfig(input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10)
    ],
)

assembler_temple = AssemblerConfig(
    name="temple",
    map_char="T",
    render_symbol="🟨",
    protocols=[
        ProtocolConfig(
            input_resources={"battery_red": 1, "ore_red": 2},
            output_resources={"laser": 1},
            cooldown=10,
        )
    ],
)


# Chest building definitions. Maybe not needed beyond the raw config?
def make_chest(
    vibe_transfers: dict[str | int, dict[str, int]] | None = None,
    initial_inventory: dict[str, int] | None = None,
    resource_limits: dict[str, int] | None = None,
    name: str | None = None,
    map_char: str = "C",
    render_symbol: str = "📦",
) -> ChestConfig:
    """Create a multi-resource chest configuration.

    Args:
        name: Name of the chest
        map_char: Character for ASCII maps
        render_symbol: Symbol for rendering
        vibe_transfers: Map from vibe to resource deltas. E.g. {'carbon': {'carbon': 10, 'energy': -5}}
        initial_inventory: Initial amounts for each resource type
        resource_limits: Maximum amount per resource (uses inventory system's built-in limits)
    """
    if vibe_transfers is None:
        # By default, deposit everything when you have a neutral expression, and withdraw specific resources when you
        # show that vibe.
        vibe_transfers = {
            "default": {"heart": 255, "carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255},
            "heart": {"heart": -1},
            "carbon": {"carbon": -10},
            "oxygen": {"oxygen": -10},
            "germanium": {"germanium": -1},
            "silicon": {"silicon": -25},
        }

    if initial_inventory is None:
        initial_inventory = {}

    if resource_limits is None:
        resource_limits = {}

    if name is None:
        name = "chest"

    return ChestConfig(
        name=name,
        map_char=map_char,
        render_symbol=render_symbol,
        vibe_transfers=vibe_transfers,
        initial_inventory=initial_inventory,
        resource_limits=resource_limits,
    )


# Example chest configurations
chest_heart = make_chest()

nav_assembler = AssemblerConfig(
    name="nav_assembler",
    map_char="_",
    render_symbol="🛣️",
    protocols=[ProtocolConfig(input_resources={}, output_resources={"heart": 1}, cooldown=255)],
)
