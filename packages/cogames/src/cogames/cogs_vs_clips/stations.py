from typing import Literal

from pydantic import Field

from cogames.cogs_vs_clips import vibes
from cogames.cogs_vs_clips.vibes import VIBE_BY_NAME
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import AssemblerConfig, ChestConfig, GridObjectConfig, ProtocolConfig, WallConfig

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


class CvCStationConfig(Config):
    start_clipped: bool = Field(default=False)
    clip_immune: bool = Field(default=False)

    def station_cfg(self) -> GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CvCWallConfig(CvCStationConfig):
    type: Literal["wall"] = Field(default="wall")

    def station_cfg(self) -> WallConfig:
        return WallConfig(name="wall", map_char="#", render_symbol=VIBE_BY_NAME["wall"].symbol)


class ExtractorConfig(CvCStationConfig):
    """Base class for all extractor configs."""

    max_uses: int = Field(default=1000)
    efficiency: int = Field(default=100)


class ChargerConfig(ExtractorConfig):
    type: Literal["charger"] = Field(default="charger")

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="charger",
            map_char="+",
            render_symbol=VIBE_BY_NAME["charger"].symbol,
            # Protocols
            allow_partial_usage=True,  # can use it while its on cooldown
            max_uses=0,  # unlimited uses
            recipes=[
                (
                    [],
                    ProtocolConfig(
                        output_resources={"energy": 50 * self.efficiency // 100},
                        cooldown=10,
                    ),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Time consuming but easy to mine.
class CarbonExtractorConfig(ExtractorConfig):
    type: Literal["carbon_extractor"] = Field(default="carbon_extractor")

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name=self.type,
            map_char="C",
            render_symbol=VIBE_BY_NAME["carbon"].symbol,
            # Protocols
            max_uses=self.max_uses,
            recipes=[
                (
                    [],
                    ProtocolConfig(
                        output_resources={"carbon": 4 * self.efficiency // 100},
                        cooldown=10,
                    ),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Accumulates oxygen over time, needs to be emptied periodically.
# Takes a lot of space, relative to usage needs.
class OxygenExtractorConfig(ExtractorConfig):
    type: Literal["oxygen_extractor"] = Field(default="oxygen_extractor")

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="oxygen_extractor",
            map_char="O",
            render_symbol=VIBE_BY_NAME["oxygen"].symbol,
            # Protocols
            max_uses=self.max_uses,
            allow_partial_usage=True,  # can use it while its on cooldown
            recipes=[
                (
                    [],
                    ProtocolConfig(
                        output_resources={"oxygen": 20},
                        cooldown=int(10_000 / self.efficiency),
                    ),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Rare and doesn't regenerate. But more cogs increase efficiency.
class GermaniumExtractorConfig(ExtractorConfig):
    type: Literal["germanium_extractor"] = Field(default="germanium_extractor")
    synergy: int = 1
    efficiency: int = 1

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="germanium_extractor",
            map_char="G",
            render_symbol=vibes.VIBE_BY_NAME["germanium"].symbol,
            # Protocols
            max_uses=1,
            recipes=[
                (
                    [],
                    ProtocolConfig(output_resources={"germanium": self.efficiency}),
                ),
                *[
                    (
                        ["germanium"] * i,
                        ProtocolConfig(output_resources={"germanium": self.efficiency + i * self.synergy}),
                    )
                    for i in range(1, 5)
                ],
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


class SiliconExtractorConfig(ExtractorConfig):
    type: Literal["silicon_extractor"] = Field(default="silicon_extractor")

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="silicon_extractor",
            map_char="S",
            render_symbol=vibes.VIBE_BY_NAME["silicon"].symbol,
            # Protocols
            max_uses=max(1, self.max_uses // 10),
            recipes=[
                (
                    [],
                    ProtocolConfig(
                        input_resources={"energy": 25},
                        output_resources={"silicon": max(1, int(25 * self.efficiency // 100))},
                    ),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


class CvCChestConfig(CvCStationConfig):
    type: Literal["communal_chest"] = Field(default="communal_chest")
    default_resource: str = Field(default="heart")

    def station_cfg(self) -> ChestConfig:
        return ChestConfig(
            name=self.type,
            map_char="C",
            render_symbol=vibes.VIBE_BY_NAME["chest"].symbol,
            resource_type=self.default_resource,
            position_deltas=[("E", 1), ("W", -1), ("N", 5), ("S", -5)],
        )


def _resource_chest(resource: str, type_id: int) -> ChestConfig:
    return ChestConfig(
        name=f"chest_{resource}",
        type_id=type_id,
        map_char="C",
        render_symbol=vibes.VIBE_BY_NAME[resource].symbol,
        resource_type=resource,
        position_deltas=[("E", 1), ("W", -1)],
    )


RESOURCE_CHESTS: dict[str, ChestConfig] = {
    "chest_carbon": _resource_chest("carbon", 118),
    "chest_oxygen": _resource_chest("oxygen", 119),
    "chest_germanium": _resource_chest("germanium", 120),
    "chest_silicon": _resource_chest("silicon", 121),
    "chest_heart": _resource_chest("heart", 122),
}


class CvCAssemblerConfig(CvCStationConfig):
    type: Literal["assembler"] = Field(default="assembler")

    def station_cfg(self) -> AssemblerConfig:
        gear = [("oxygen", "modulator"), ("germanium", "scrambler"), ("silicon", "resonator"), ("carbon", "decoder")]
        return AssemblerConfig(
            name=self.type,
            map_char="&",
            render_symbol=vibes.VIBE_BY_NAME["assembler"].symbol,
            clip_immune=True,
            recipes=[
                (
                    ["heart"] * (i + 1),
                    ProtocolConfig(
                        input_resources={"carbon": 20, "oxygen": 20, "germanium": 5 - i, "silicon": 50, "energy": 20},
                        output_resources={"heart": 1},
                    ),
                )
                for i in range(4)
            ]
            + [
                (
                    ["gear", gear[i][0]],
                    ProtocolConfig(input_resources={gear[i][0]: 1}, output_resources={gear[i][1]: 1}),
                )
                for i in range(len(gear))
            ],
        )
