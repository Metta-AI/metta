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
    "battery_red",
    "resonator",
    "scrambler",
]


def charger(max_use: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="charger",
        type_id=5,
        map_char="H",
        render_symbol="⚡",
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
        map_char="N",
        render_symbol="⚫",
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
        map_char="O",
        render_symbol="🔵",
        allow_partial_usage=True,  # can use it while its on cooldown
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
        map_char="E",
        render_symbol="🟣",
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
        map_char="I",
        render_symbol="🔷",
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
        map_char="V",
        render_symbol="🔹",
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
        map_char="Y",
        render_symbol="🟪",
        recipes=[
            (["Any"], RecipeConfig(output_resources={"germanium": 1}, cooldown=1, max_use=5)),
        ],
    )


def oxygen_ex_dep() -> ConverterConfig:
    return ConverterConfig(
        name="oxygen_ex_dep",
        type_id=18,
        map_char="Q",
        render_symbol="⬜",
        output_resources={"oxygen": 1},
        max_output=10,
        cooldown=10,
    )


def carbon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_ex_dep",
        type_id=19,
        map_char="K",
        render_symbol="⬛",
        recipes=[
            (["Any"], RecipeConfig(output_resources={"carbon": 1}, cooldown=1, max_use=5)),
        ],
    )


def chest() -> ChestConfig:
    return ChestConfig(
        name="chest",
        type_id=17,
        map_char="C",
        render_symbol="📦",
        resource_type="heart",
        deposit_positions=["E"],
        withdrawal_positions=["W"],
    )


def assembler() -> AssemblerConfig:
    return AssemblerConfig(
        name="assembler",
        type_id=8,
        map_char="Z",
        render_symbol="🔄",
        recipes=[
            (
                ["E"],
                RecipeConfig(
                    input_resources={"energy": 3},
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            ),
            (
                ["N"],
                RecipeConfig(
                    input_resources={"germanium": 1},
                    output_resources={"decoder": 1},
                    cooldown=1,
                ),
            ),
            (
                ["S"],
                RecipeConfig(
                    input_resources={"carbon": 3},
                    output_resources={"modulator": 1},
                    cooldown=1,
                ),
            ),
            (
                ["W"],
                RecipeConfig(
                    input_resources={"oxygen": 3},
                    output_resources={"scrambler": 1},
                    cooldown=1,
                ),
            ),
        ],
    )
