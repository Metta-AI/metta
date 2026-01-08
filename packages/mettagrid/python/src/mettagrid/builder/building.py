from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ProtocolConfig,
    WallConfig,
)

wall = WallConfig(name="wall", render_symbol="⬛")

# Assembler building definitions
assembler_assembler = AssemblerConfig(
    name="assembler",
    render_symbol="🎯",
    protocols=[ProtocolConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10)],
)


def make_assembler_mine(color: str) -> AssemblerConfig:
    symbol_map = {"red": "🔺", "blue": "🔷", "green": "💚"}
    return AssemblerConfig(
        name=f"mine_{color}",
        render_symbol=symbol_map[color],
        protocols=[ProtocolConfig(output_resources={f"ore_{color}": 1}, cooldown=50)],
    )


assembler_mine_red = make_assembler_mine("red")
assembler_mine_blue = make_assembler_mine("blue")
assembler_mine_green = make_assembler_mine("green")


def make_assembler_generator(color: str) -> AssemblerConfig:
    symbol_map = {"red": "🔋", "blue": "🔌", "green": "🟢"}
    return AssemblerConfig(
        name=f"generator_{color}",
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
    render_symbol="🟥",
    protocols=[
        ProtocolConfig(input_resources={"battery_red": 1, "ore_red": 2}, output_resources={"laser": 1}, cooldown=10)
    ],
)

assembler_armory = AssemblerConfig(
    name="armory",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={"ore_red": 3}, output_resources={"armor": 1}, cooldown=10)],
)

nav_assembler = AssemblerConfig(
    name="nav_assembler",
    render_symbol="🛣️",
    protocols=[ProtocolConfig(input_resources={}, output_resources={"heart": 1}, cooldown=255)],
)
