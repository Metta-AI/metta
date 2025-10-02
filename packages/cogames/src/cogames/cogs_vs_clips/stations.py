from typing import Optional

from cogames.cogs_vs_clips import protocols
from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig

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
    recipe = protocols.standard_charging_recipe()
    return AssemblerConfig(
        name="charger",
        type_id=5,
        map_char="+",
        render_symbol="⚡",
        allow_partial_usage=True,  # can use it while its on cooldown
        max_uses=max_uses or 0,
        recipes=protocols.protocol(recipe),
    )


# Time consuming but easy to mine.
def carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        type_id=2,
        map_char="C",
        render_symbol="⚫",
        max_uses=max_uses or 0,
        recipes=protocols.protocol(protocols.standard_carbon_recipe()),
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
        recipes=protocols.protocol(protocols.standard_oxygen_recipe()),
    )


# Need little, takes a long time to regen.
def germanium_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_extractor",
        type_id=4,
        map_char="G",
        render_symbol="🟣",
        max_uses=max_uses or 2,
        recipes=(
            protocols.protocol(protocols.germanium_recipe(1), num_agents=1)
            + protocols.protocol(protocols.germanium_recipe(2), num_agents=2)
            + protocols.protocol(protocols.germanium_recipe(3), num_agents=3)
            + protocols.protocol(protocols.germanium_recipe(4), min_agents=4)
        ),
    )


# Plentiful but requires energy / work and need a lot.
def silicon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor",
        type_id=15,
        map_char="S",
        render_symbol="🔷",
        max_uses=max_uses or 0,
        recipes=protocols.protocol(protocols.standard_silicon_recipe()),
    )


def carbon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_ex_dep",
        type_id=19,
        map_char="c",
        render_symbol="⬛",
        max_uses=100,
        recipes=protocols.protocol(protocols.low_carbon_recipe()),
    )


def oxygen_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_ex_dep",
        type_id=18,
        map_char="o",  # lowercase o for depleted oxygen
        render_symbol="⬜",
        max_uses=10,
        allow_partial_usage=True,
        recipes=protocols.protocol(protocols.low_oxygen_recipe()),
    )


def germanium_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_ex_dep",
        type_id=20,
        map_char="g",
        render_symbol="🟪",
        max_uses=1,
        recipes=(
            protocols.protocol(protocols.germanium_recipe(1), num_agents=1)
            + protocols.protocol(protocols.germanium_recipe(2), num_agents=2)
            + protocols.protocol(protocols.germanium_recipe(3), num_agents=3)
            + protocols.protocol(protocols.germanium_recipe(4), min_agents=4)
        ),
    )


def silicon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_ex_dep",
        type_id=16,
        map_char="s",
        render_symbol="🔹",
        max_uses=10,
        recipes=protocols.protocol(protocols.low_silicon_recipe()),
    )


def chest() -> ChestConfig:
    return ChestConfig(
        name="chest",
        type_id=17,
        map_char="=",
        render_symbol="📦",
        resource_type="heart",
        position_deltas=[("E", 1), ("W", -1)],
    )


def chest_carbon() -> ChestConfig:
    return ChestConfig(
        name="chest_carbon",
        type_id=31,
        map_char="L",
        render_symbol="📦",
        resource_type="carbon",
        position_deltas=[("E", 1), ("W", -1), ("N", 5), ("S", -5)],
    )


def chest_oxygen() -> ChestConfig:
    return ChestConfig(
        name="chest_oxygen",
        type_id=32,
        map_char="M",
        render_symbol="📦",
        resource_type="oxygen",
        position_deltas=[("E", 1), ("W", -1), ("N", 10), ("S", -10)],
    )


def chest_germanium() -> ChestConfig:
    return ChestConfig(
        name="chest_germanium",
        type_id=33,
        map_char="N",
        render_symbol="📦",
        resource_type="germanium",
        position_deltas=[("E", 1), ("W", -1), ("N", 5), ("S", -5)],
    )


def chest_silicon() -> ChestConfig:
    return ChestConfig(
        name="chest_silicon",
        type_id=34,
        map_char="O",
        render_symbol="📦",
        resource_type="silicon",
        position_deltas=[("E", 1), ("W", -1), ("N", 25), ("S", -25)],
    )


def assembler() -> AssemblerConfig:
    return AssemblerConfig(
        name="assembler",
        type_id=8,
        map_char="&",
        render_symbol="🔄",
        recipes=[
            (["N"], protocols.one_agent_heart_recipe()),
            (["W"], protocols.one_agent_heart_recipe()),
            (["S"], protocols.one_agent_heart_recipe()),
            (["E"], protocols.one_agent_heart_recipe()),
            (["N", "E"], protocols.two_agent_heart_recipe()),
            (["N", "W"], protocols.two_agent_heart_recipe()),
            (["N", "S"], protocols.two_agent_heart_recipe()),
            (["E", "S"], protocols.two_agent_heart_recipe()),
            (["E", "W"], protocols.two_agent_heart_recipe()),
            (["S", "W"], protocols.two_agent_heart_recipe()),
            (["N", "E", "W"], protocols.three_agent_heart_recipe()),
            (["N", "E", "S"], protocols.three_agent_heart_recipe()),
            (["N", "W", "S"], protocols.three_agent_heart_recipe()),
            (["E", "W", "S"], protocols.three_agent_heart_recipe()),
            (["N", "E", "W", "S"], protocols.four_agent_heart_recipe()),
        ],
        # (
        #     ["E"],
        #     RecipeConfig(
        #         input_resources={"energy": 3},
        #         output_resources={"heart": 1},
        #         cooldown=1,
        #     ),
        # ),
        # (
        #     ["N"],
        #     RecipeConfig(
        #         input_resources={"germanium": 1},
        #         output_resources={"decoder": 1},
        #         cooldown=1,
        #     ),
        # ),
        # (
        #     ["S"],
        #     RecipeConfig(
        #         input_resources={"carbon": 3},
        #         output_resources={"modulator": 1},
        #         cooldown=1,
        #     ),
        # ),
        # (
        #     ["W"],
        #     RecipeConfig(
        #         input_resources={"oxygen": 3},
        #         output_resources={"scrambler": 1},
        #         cooldown=1,
        #     ),
        # ),
    )
