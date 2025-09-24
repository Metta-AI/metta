from mettagrid.config.mettagrid_config import AssemblerConfig, RecipeConfig

resources = [
    "energy",
    "carbon",
    "oxygen",
    "geranium",
    "silicon",
    "heart",
    "disruptor",
    "modulator",
    "resonator",
    "scrabbler",
]


def charger() -> AssemblerConfig:
    return AssemblerConfig(
        name="charger",
        type_id=5,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"energy": 100},
                    cooldown=1,
                ),
            )
        ],
    )


def carbon_extractor() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        type_id=2,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 1},
                    output_resources={"carbon": 1},
                    cooldown=1,
                ),
            )
        ],
    )


def oxygen_extractor() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_extractor",
        type_id=3,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 1},
                    output_resources={"oxygen": 10},
                    cooldown=1,
                ),
            )
        ],
    )


def geranium_extractor() -> AssemblerConfig:
    return AssemblerConfig(
        name="geranium_extractor",
        type_id=4,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 1},
                    output_resources={"geranium": 1},
                    cooldown=1,
                ),
            )
        ],
    )


def silicon_extractor() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor",
        type_id=15,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 10},
                    output_resources={"silicon": 1},
                    cooldown=1,
                ),
            )
        ],
    )


def chest() -> AssemblerConfig:
    return AssemblerConfig(
        name="chest",
        type_id=17,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"heart": 1},
                    output_resources={},
                    cooldown=1,
                ),
            )
        ],
    )


def assembler() -> AssemblerConfig:
    return AssemblerConfig(
        name="assembler",
        type_id=8,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 3},
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )
        ],
    )
