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


def clipped_carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_carbon_extractor",
        type_id=35,
        map_char="B",
        render_symbol="⚫",
        max_uses=max_uses or 0,
        start_clipped=True,
        recipes=protocols.protocol(protocols.standard_carbon_recipe()),
    )


def clipped_oxygen_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_oxygen_extractor",
        type_id=36,
        map_char="N",
        render_symbol="🔵",
        max_uses=max_uses or 0,
        start_clipped=True,
        recipes=protocols.protocol(protocols.standard_carbon_recipe()),
    )


def clipped_germanium_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_germanium_extractor",
        type_id=37,
        map_char="F",
        render_symbol="🟣",
        max_uses=max_uses or 2,
        start_clipped=True,
        recipes=protocols.protocol(protocols.germanium_recipe(1), num_agents=1)
        + protocols.protocol(protocols.germanium_recipe(2), num_agents=2)
        + protocols.protocol(protocols.germanium_recipe(3), num_agents=3)
        + protocols.protocol(protocols.germanium_recipe(4), min_agents=4),
    )


def clipped_silicon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_silicon_extractor",
        type_id=38,
        map_char="R",
        render_symbol="🔷",
        max_uses=max_uses or 0,
        start_clipped=True,
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


# Chest characters are the letter after the relative resource type.
def chest_carbon() -> ChestConfig:
    return ChestConfig(
        name="chest_carbon",
        type_id=31,
        map_char="D",
        render_symbol="📦",
        resource_type="carbon",
        initial_inventory=50,
        position_deltas=[("E", 1), ("W", -1), ("N", 5), ("S", -5)],
    )


def chest_oxygen() -> ChestConfig:
    return ChestConfig(
        name="chest_oxygen",
        type_id=32,
        map_char="P",
        render_symbol="📦",
        resource_type="oxygen",
        initial_inventory=50,
        position_deltas=[("E", 1), ("W", -1), ("N", 10), ("S", -10)],
    )


def chest_germanium() -> ChestConfig:
    return ChestConfig(
        name="chest_germanium",
        type_id=33,
        map_char="H",
        render_symbol="📦",
        resource_type="germanium",
        initial_inventory=5,
        position_deltas=[("E", 1), ("W", -1), ("N", 5), ("S", -5)],
    )


def chest_silicon() -> ChestConfig:
    return ChestConfig(
        name="chest_silicon",
        type_id=34,
        map_char="T",
        render_symbol="📦",
        resource_type="silicon",
        initial_inventory=100,
        position_deltas=[("E", 1), ("W", -1), ("N", 25), ("S", -25)],
    )


def assembler() -> AssemblerConfig:
    return AssemblerConfig(
        name="assembler",
        type_id=8,
        map_char="&",
        render_symbol="🔄",
        clip_immune=True,
        recipes=[
            # Single agent heart recipes (cardinal directions)
            (["N"], protocols.one_agent_heart_recipe()),
            (["W"], protocols.one_agent_heart_recipe()),
            (["S"], protocols.one_agent_heart_recipe()),
            (["E"], protocols.one_agent_heart_recipe()),
            # Two agent heart recipes (two cardinal directions)
            (["N", "E"], protocols.two_agent_heart_recipe()),
            (["N", "W"], protocols.two_agent_heart_recipe()),
            (["N", "S"], protocols.two_agent_heart_recipe()),
            (["E", "S"], protocols.two_agent_heart_recipe()),
            (["E", "W"], protocols.two_agent_heart_recipe()),
            (["S", "W"], protocols.two_agent_heart_recipe()),
            # Three agent heart recipes
            (["N", "E", "W"], protocols.three_agent_heart_recipe()),
            (["N", "E", "S"], protocols.three_agent_heart_recipe()),
            (["N", "W", "S"], protocols.three_agent_heart_recipe()),
            (["E", "W", "S"], protocols.three_agent_heart_recipe()),
            # Four agent heart recipe
            (["N", "E", "W", "S"], protocols.four_agent_heart_recipe()),
            # Equipment recipes: NE diagonal + cardinal = decoder
            (["NE", "N"], protocols.decoder_recipe()),
            (["NE", "E"], protocols.decoder_recipe()),
            (["NE", "S"], protocols.decoder_recipe()),
            (["NE", "W"], protocols.decoder_recipe()),
            # Equipment recipes: SE diagonal + cardinal = modulator
            (["SE", "N"], protocols.modulator_recipe()),
            (["SE", "E"], protocols.modulator_recipe()),
            (["SE", "S"], protocols.modulator_recipe()),
            (["SE", "W"], protocols.modulator_recipe()),
            # Equipment recipes: SW diagonal + cardinal = scrambler
            (["SW", "N"], protocols.scrambler_recipe()),
            (["SW", "E"], protocols.scrambler_recipe()),
            (["SW", "S"], protocols.scrambler_recipe()),
            (["SW", "W"], protocols.scrambler_recipe()),
            # Equipment recipes: NW diagonal + cardinal = resonator
            (["NW", "N"], protocols.resonator_recipe()),
            (["NW", "E"], protocols.resonator_recipe()),
            (["NW", "S"], protocols.resonator_recipe()),
            (["NW", "W"], protocols.resonator_recipe()),
        ],
    )
