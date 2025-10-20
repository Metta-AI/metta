from mettagrid.config.mettagrid_config import AssemblerConfig, ConverterConfig, WallConfig, RecipeConfig

wall = WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛")
block = WallConfig(name="block", type_id=14, map_char="s", render_symbol="📦", swappable=True)

mine_red = AssemblerConfig(
    name="mine_red",
    type_id=2,
    map_char="m",
    render_symbol="🔺",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

mine_blue = AssemblerConfig(
    name="mine_blue",
    type_id=3,
    map_char="b",
    render_symbol="🔷",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

mine_green = AssemblerConfig(
    name="mine_green",
    type_id=4,
    map_char="g",
    render_symbol="💚",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

generator_red = AssemblerConfig(
    name="generator_red",
    type_id=5,
    map_char="n",
    render_symbol="🔋",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

generator_blue = AssemblerConfig(
    name="generator_blue",
    type_id=6,
    map_char="B",
    render_symbol="🔌",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

generator_green = AssemblerConfig(
    name="generator_green",
    type_id=7,
    map_char="G",
    render_symbol="🟢",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)

altar = AssemblerConfig(
    name="altar",
    type_id=8,
    map_char="_",
    render_symbol="🎯",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)


lasery = AssemblerConfig(
    name="lasery",
    type_id=15,
    map_char="S",
    render_symbol="🟥",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)
armory = AssemblerConfig(
    name="armory",
    type_id=16,
    map_char="o",
    render_symbol="🔵",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)
lab = AssemblerConfig(
    name="lab",
    type_id=17,
    map_char="L",
    render_symbol="🔵",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)
factory = AssemblerConfig(
    name="factory",
    type_id=18,
    map_char="F",
    render_symbol="🟪",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)
temple = AssemblerConfig(
    name="temple",
    type_id=19,
    map_char="T",
    render_symbol="🟨",
    recipes=[([], RecipeConfig(input_resources={}, output_resources={}, cooldown=5))],
)
