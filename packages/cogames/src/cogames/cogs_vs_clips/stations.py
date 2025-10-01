from typing import Optional

from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, ConverterConfig, RecipeConfig

resources = [
    "energy",
    "carbon",
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "decoder",
    "modulator",
    "resonator",
    "scrambler",
]


def charger(max_use: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="charger",
        type_id=5,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"energy": 50},
                    cooldown=1,
                    max_use=max_use,
                ),
            )
        ],
    )


# rare but easy to mine
def carbon_extractor(max_use: Optional[int] = 1) -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        type_id=2,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 4},
                    output_resources={"carbon": 25},
                    max_use=max_use,
                ),
            )
        ],
    )


# accumulates oxygen over time, needs to be emptied periodically
def oxygen_extractor(max_use: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_extractor",
        type_id=3,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"oxygen": 1},
                    max_use=max_use,
                ),
            )
        ],
    )


# need little, takes a long time to regen
def germanium_extractor(max_use: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_extractor",
        type_id=4,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"germanium": 1},
                    cooldown=250,
                    max_use=max_use,
                ),
            )
        ],
    )


# plentiful but requires energy / work and need a lot
def silicon_extractor(max_use: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor",
        type_id=15,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 25},
                    output_resources={"silicon": 25},
                    cooldown=1,
                    max_use=max_use,
                ),
            )
        ],
    )


def silicon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_ex_dep",
        type_id=16,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"silicon": 1},
                    cooldown=1,
                    max_use=5,
                ),
            )
        ],
    )


def germanium_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_ex_dep",
        type_id=20,
        recipes=[
            (["Any"], RecipeConfig(output_resources={"germanium": 1}, cooldown=1, max_use=5)),
        ],
    )


def oxygen_ex_dep() -> ConverterConfig:
    return ConverterConfig(
        name="oxygen_ex_dep",
        type_id=18,
        output_resources={"oxygen": 1},
        max_output=10,
        cooldown=10,
    )


def carbon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_ex_dep",
        type_id=19,
        recipes=[
            (["Any"], RecipeConfig(output_resources={"carbon": 1}, cooldown=1, max_use=5)),
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
