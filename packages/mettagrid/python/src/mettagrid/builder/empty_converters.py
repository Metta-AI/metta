from mettagrid.config.mettagrid_config import ConverterConfig, WallConfig

wall = WallConfig(type_id=1)
block = WallConfig(type_id=14, swappable=True)

mine_red = ConverterConfig(
    type_id=2,
    input_resources={},
    output_resources={},
    cooldown=5,
)

mine_blue = ConverterConfig(
    type_id=3,
    input_resources={},
    output_resources={},
    cooldown=5,
)

mine_green = ConverterConfig(
    type_id=4,
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_red = ConverterConfig(
    type_id=5,
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_blue = ConverterConfig(
    type_id=6,
    input_resources={},
    output_resources={},
    cooldown=5,
)

generator_green = ConverterConfig(
    type_id=7,
    input_resources={},
    output_resources={},
    cooldown=5,
)

altar = ConverterConfig(
    type_id=8,
    input_resources={},
    output_resources={},
    cooldown=5,
)


lasery = ConverterConfig(
    type_id=15,
    input_resources={},
    output_resources={},
    cooldown=5,
)
armory = ConverterConfig(
    type_id=16,
    input_resources={},
    output_resources={},
    cooldown=5,
)
lab = ConverterConfig(
    type_id=17,
    input_resources={},
    output_resources={},
    cooldown=5,
)
factory = ConverterConfig(
    type_id=18,
    input_resources={},
    output_resources={},
    cooldown=5,
)
temple = ConverterConfig(
    type_id=19,
    input_resources={},
    output_resources={},
    cooldown=5,
)
