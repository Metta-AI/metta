from typing import Optional

from cogames.cogs_vs_clips import protocols
from cogames.cogs_vs_clips.glyphs import GLYPH_NAMES
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
        count_based_recipes=[recipe] * 9,
    )


# Time consuming but easy to mine.
def carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_extractor",
        type_id=2,
        map_char="C",
        render_symbol="⚫",
        max_uses=max_uses or 0,
        count_based_recipes=[protocols.standard_carbon_recipe()] * 9,
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
        count_based_recipes=[protocols.standard_oxygen_recipe()] * 9,
    )


# Need little, takes a long time to regen.
def germanium_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_extractor",
        type_id=4,
        map_char="G",
        render_symbol="🟣",
        max_uses=max_uses or 2,
        count_based_recipes=[protocols.germanium_recipe(1)] * 9,
    )


# Plentiful but requires energy / work and need a lot.
def silicon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_extractor",
        type_id=15,
        map_char="S",
        render_symbol="🔷",
        max_uses=max_uses or 0,
        count_based_recipes=[protocols.standard_silicon_recipe()] * 9,
    )


def clipped_carbon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_carbon_extractor",
        type_id=35,
        map_char="B",
        render_symbol="⚫",
        max_uses=max_uses or 0,
        start_clipped=True,
        count_based_recipes=[protocols.standard_carbon_recipe()] * 9,
    )


def clipped_oxygen_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_oxygen_extractor",
        type_id=36,
        map_char="N",
        render_symbol="🔵",
        max_uses=max_uses or 0,
        start_clipped=True,
        count_based_recipes=[protocols.standard_oxygen_recipe()] * 9,
    )


def clipped_germanium_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_germanium_extractor",
        type_id=37,
        map_char="F",
        render_symbol="🟣",
        max_uses=max_uses or 2,
        start_clipped=True,
        count_based_recipes=[protocols.germanium_recipe(1)] * 9,
    )


def clipped_silicon_extractor(max_uses: Optional[int] = None) -> AssemblerConfig:
    return AssemblerConfig(
        name="clipped_silicon_extractor",
        type_id=38,
        map_char="R",
        render_symbol="🔷",
        max_uses=max_uses or 0,
        start_clipped=True,
        count_based_recipes=[protocols.standard_silicon_recipe()] * 9,
    )


def carbon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="carbon_ex_dep",
        type_id=19,
        map_char="c",
        render_symbol="⬛",
        max_uses=100,
        count_based_recipes=[protocols.low_carbon_recipe()] * 9,
    )


def oxygen_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="oxygen_ex_dep",
        type_id=18,
        map_char="o",  # lowercase o for depleted oxygen
        render_symbol="⬜",
        max_uses=10,
        allow_partial_usage=True,
        count_based_recipes=[protocols.low_oxygen_recipe()] * 9,
    )


def germanium_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="germanium_ex_dep",
        type_id=20,
        map_char="g",
        render_symbol="🟪",
        max_uses=1,
        count_based_recipes=[protocols.germanium_recipe(1)] * 9,
    )


def silicon_ex_dep() -> AssemblerConfig:
    return AssemblerConfig(
        name="silicon_ex_dep",
        type_id=16,
        map_char="s",
        render_symbol="🔹",
        max_uses=10,
        count_based_recipes=[protocols.low_silicon_recipe()] * 9,
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
        vibe_recipes=[
            ([GLYPH_NAMES["heart"]], protocols.one_agent_heart_recipe()),
            ([GLYPH_NAMES["heart"], GLYPH_NAMES["heart"]], protocols.two_agent_heart_recipe()),
            ([GLYPH_NAMES["heart"], GLYPH_NAMES["heart"], GLYPH_NAMES["heart"]], protocols.three_agent_heart_recipe()),
            (
                [GLYPH_NAMES["heart"], GLYPH_NAMES["heart"], GLYPH_NAMES["heart"], GLYPH_NAMES["heart"]],
                protocols.four_agent_heart_recipe(),
            ),
            ([GLYPH_NAMES["angry"], GLYPH_NAMES["carbon"]], protocols.decoder_recipe()),
            ([GLYPH_NAMES["angry"], GLYPH_NAMES["oxygen"]], protocols.modulator_recipe()),
            ([GLYPH_NAMES["angry"], GLYPH_NAMES["germanium"]], protocols.scrambler_recipe()),
            ([GLYPH_NAMES["angry"], GLYPH_NAMES["silicon"]], protocols.resonator_recipe()),
        ],
    )
