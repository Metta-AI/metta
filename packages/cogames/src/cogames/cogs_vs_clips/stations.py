from typing import Optional

from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, RecipeConfig

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


def charger(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="charger",
        type_id=5,
        map_char="H",
        render_symbol="⚡",
        max_uses=max_uses or 0,
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


# Time consuming but easy to mine.
def carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        type_id=2,
        map_char="N",
        render_symbol="⚫",
        max_uses=max_uses or 0,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"carbon": 5},
                ),
            )
        ],
    )


# Accumulates oxygen over time, needs to be emptied periodically. Takes a lot of space, relative to usage needs.
def oxygen_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_extractor",
        type_id=3,
        map_char="O",
        render_symbol="🔵",
        allow_partial_usage=True,  # can use it while its on cooldown
        max_uses=max_uses or 0,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"oxygen": 100},
                    cooldown=100,
                ),
            )
        ],
    )


# Need little, takes a long time to regen.
def germanium_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_extractor",
        type_id=4,
        map_char="E",
        render_symbol="🟣",
        max_uses=max_uses or 0,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    output_resources={"germanium": 1},
                    cooldown=250,
                ),
            )
        ],
    )


# Plentiful but requires energy / work and need a lot.
def silicon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor",
        type_id=15,
        map_char="I",
        render_symbol="🔷",
        max_uses=max_uses or 0,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 25},
                    output_resources={"silicon": 25},
                    cooldown=1,
                ),
            )
        ],
    )


def carbon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_ex_dep",
        type_id=19,
        map_char="K",
        render_symbol="⬛",
        max_uses=50,
        recipes=[
            (["Any"], RecipeConfig(output_resources={"carbon": 1}, cooldown=1)),
        ],
    )


def oxygen_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_ex_dep",
        type_id=18,
        map_char="Q",
        render_symbol="⬜",
        max_uses=5,
        allow_partial_usage=True,
        recipes=[
            (["Any"], RecipeConfig(output_resources={"oxygen": 20}, cooldown=20)),
        ],
    )


def germanium_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_ex_dep",
        type_id=20,
        map_char="Y",
        render_symbol="🟪",
        max_uses=5,
        recipes=[
            (["Any"], RecipeConfig(output_resources={"germanium": 1}, cooldown=1)),
        ],
    )


def silicon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_ex_dep",
        type_id=16,
        map_char="V",
        render_symbol="🔹",
        max_uses=5,
        recipes=[
            (
                ["Any"],
                RecipeConfig(
                    input_resources={"energy": 25},
                    output_resources={"silicon": 10},
                    cooldown=1,
                ),
            )
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
