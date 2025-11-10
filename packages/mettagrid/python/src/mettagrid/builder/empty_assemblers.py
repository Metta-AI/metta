from mettagrid.config.mettagrid_config import AssemblerConfig, ProtocolConfig, WallConfig

wall = WallConfig(name="wall", map_char="#", render_symbol="⬛")
block = WallConfig(name="block", map_char="s", render_symbol="📦", swappable=True)

mine_red = AssemblerConfig(
    name="mine_red",
    map_char="m",
    render_symbol="🔺",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

mine_blue = AssemblerConfig(
    name="mine_blue",
    map_char="b",
    render_symbol="🔷",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

mine_green = AssemblerConfig(
    name="mine_green",
    map_char="g",
    render_symbol="💚",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_red = AssemblerConfig(
    name="generator_red",
    map_char="n",
    render_symbol="🔋",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_blue = AssemblerConfig(
    name="generator_blue",
    map_char="B",
    render_symbol="🔌",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

generator_green = AssemblerConfig(
    name="generator_green",
    map_char="G",
    render_symbol="🟢",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)

altar = AssemblerConfig(
    name="altar",
    map_char="_",
    render_symbol="🎯",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)


lasery = AssemblerConfig(
    name="lasery",
    map_char="S",
    render_symbol="🟥",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
armory = AssemblerConfig(
    name="armory",
    map_char="o",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
lab = AssemblerConfig(
    name="lab",
    map_char="L",
    render_symbol="🔵",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
factory = AssemblerConfig(
    name="factory",
    map_char="F",
    render_symbol="🟪",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
temple = AssemblerConfig(
    name="temple",
    map_char="T",
    render_symbol="🟨",
    protocols=[ProtocolConfig(input_resources={}, output_resources={}, cooldown=5)],
)
