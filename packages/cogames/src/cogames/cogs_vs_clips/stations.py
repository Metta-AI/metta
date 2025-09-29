from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, RecipeConfig

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
                    output_resources={"energy": 50},
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
                    output_resources={"geranium": 10},
                    cooldown=100,
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


# --- Variants: fast (low cooldown, low yield) and slow (high cooldown, high yield)


def carbon_extractor_fast() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor_fast",
        type_id=18,
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


def carbon_extractor_slow() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor_slow",
        type_id=19,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 3},
                    output_resources={"carbon": 5},
                    cooldown=10,
                ),
            )
        ],
    )


def oxygen_extractor_fast() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_extractor_fast",
        type_id=20,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 1},
                    output_resources={"oxygen": 5},
                    cooldown=1,
                ),
            )
        ],
    )


def oxygen_extractor_slow() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_extractor_slow",
        type_id=21,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 3},
                    output_resources={"oxygen": 20},
                    cooldown=10,
                ),
            )
        ],
    )


def geranium_extractor_fast() -> AssemblerConfig:
    return AssemblerConfig(
        name="geranium_extractor_fast",
        type_id=22,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 2},
                    output_resources={"geranium": 2},
                    cooldown=5,
                ),
            )
        ],
    )


def geranium_extractor_slow() -> AssemblerConfig:
    return AssemblerConfig(
        name="geranium_extractor_slow",
        type_id=23,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 5},
                    output_resources={"geranium": 20},
                    cooldown=200,
                ),
            )
        ],
    )


def silicon_extractor_fast() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor_fast",
        type_id=24,
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


def silicon_extractor_slow() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor_slow",
        type_id=25,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 15},
                    output_resources={"silicon": 5},
                    cooldown=10,
                ),
            )
        ],
    )


def chest() -> ChestConfig:
    return ChestConfig(
        type_id=17,
        resource_type="heart",
        deposit_positions=["E"],
        withdrawal_positions=["W"],
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
