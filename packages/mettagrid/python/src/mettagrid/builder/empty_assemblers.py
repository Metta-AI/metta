from mettagrid.config.mettagrid_config import AssemblerConfig, ProtocolConfig, WallConfig

wall = WallConfig(name="wall", render_symbol="⬛")

mine_red = AssemblerConfig(
    name="mine_red",
    render_symbol="🔺",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

mine_blue = AssemblerConfig(
    name="mine_blue",
    render_symbol="🔷",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

mine_green = AssemblerConfig(
    name="mine_green",
    render_symbol="💚",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_red = AssemblerConfig(
    name="generator_red",
    render_symbol="🔋",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_blue = AssemblerConfig(
    name="generator_blue",
    render_symbol="🔌",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_green = AssemblerConfig(
    name="generator_green",
    render_symbol="🟢",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

assembler = AssemblerConfig(
    name="assembler",
    render_symbol="🎯",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)


lasery = AssemblerConfig(
    name="lasery",
    render_symbol="🟥",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
armory = AssemblerConfig(
    name="armory",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
lab = AssemblerConfig(
    name="lab",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
factory = AssemblerConfig(
    name="factory",
    render_symbol="🟪",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
temple = AssemblerConfig(
    name="temple",
    render_symbol="🟨",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
