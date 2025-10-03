from mettagrid.config.mettagrid_config import ConverterConfig, WallConfig

wall = WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛")
block = WallConfig(name="block", type_id=14, map_char="s", render_symbol="📦", swappable=True)

mine_red = ConverterConfig(
    name="mine_red",
    type_id=2,
    map_char="m",
    render_symbol="🔺",
    input_resources={},
    output_resources={},
    cooldown=5,
)

mine_blue = ConverterConfig(
    name="mine_blue",
    type_id=3,
    map_char="b",
    render_symbol="🔷",
    input_resources={},
    output_resources={},
    cooldown=5,
)

mine_green = ConverterConfig(
    name="mine_green",
    type_id=4,
    map_char="g",
    render_symbol="💚",
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_red = ConverterConfig(
    name="generator_red",
    type_id=5,
    map_char="n",
    render_symbol="🔋",
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_blue = ConverterConfig(
    name="generator_blue",
    type_id=6,
    map_char="B",
    render_symbol="🔌",
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_green = ConverterConfig(
    name="generator_green",
    type_id=7,
    map_char="G",
    render_symbol="🟢",
    input_resources={},
    output_resources={},
    cooldown=5,
)

altar = ConverterConfig(
    name="altar",
    type_id=8,
    map_char="_",
    render_symbol="🎯",
    input_resources={},
    output_resources={},
    cooldown=5,
)


lasery = ConverterConfig(
    name="lasery",
    type_id=15,
    map_char="S",
    render_symbol="🟥",
    input_resources={},
    output_resources={},
    cooldown=5,
)
armory = ConverterConfig(
    name="armory",
    type_id=16,
    map_char="o",
    render_symbol="🔵",
    input_resources={},
    output_resources={},
    cooldown=5,
)
lab = ConverterConfig(
    name="lab",
    type_id=17,
    map_char="L",
    render_symbol="🔵",
    input_resources={},
    output_resources={},
    cooldown=5,
)
factory = ConverterConfig(
    name="factory",
    type_id=18,
    map_char="F",
    render_symbol="🟪",
    input_resources={},
    output_resources={},
    cooldown=5,
)
temple = ConverterConfig(
    name="temple",
    type_id=19,
    map_char="T",
    render_symbol="🟨",
    input_resources={},
    output_resources={},
    cooldown=5,
)
