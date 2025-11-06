import typing

import pydantic

import cogames.cogs_vs_clips
import mettagrid.base_config
import mettagrid.config.mettagrid_config

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


class CvCStationConfig(mettagrid.base_config.Config):
    start_clipped: bool = pydantic.Field(default=False)
    clip_immune: bool = pydantic.Field(default=False)

    def station_cfg(self) -> mettagrid.config.mettagrid_config.GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CvCWallConfig(CvCStationConfig):
    type: typing.Literal["wall"] = pydantic.Field(default="wall")

    def station_cfg(self) -> mettagrid.config.mettagrid_config.WallConfig:
        return mettagrid.config.mettagrid_config.WallConfig(
            name="wall", map_char="#", render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["wall"].symbol
        )


class ExtractorConfig(CvCStationConfig):
    """Base class for all extractor configs."""

    max_uses: int = pydantic.Field(default=1000)
    efficiency: int = pydantic.Field(default=100)


class ChargerConfig(ExtractorConfig):
    type: typing.Literal["charger"] = pydantic.Field(default="charger")

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name="charger",
            map_char="+",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["charger"].symbol,
            # Protocols
            allow_partial_usage=True,  # can use it while its on cooldown
            max_uses=0,  # unlimited uses
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    output_resources={"energy": 50 * self.efficiency // 100},
                    cooldown=10,
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Time consuming but easy to mine.
class CarbonExtractorConfig(ExtractorConfig):
    type: typing.Literal["carbon_extractor"] = pydantic.Field(default="carbon_extractor")

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name=self.type,
            map_char="C",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["carbon"].symbol,
            # Protocols
            max_uses=self.max_uses,
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    output_resources={"carbon": 4 * self.efficiency // 100},
                    cooldown=0,
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Accumulates oxygen over time, needs to be emptied periodically.
# Takes a lot of space, relative to usage needs.
class OxygenExtractorConfig(ExtractorConfig):
    type: typing.Literal["oxygen_extractor"] = pydantic.Field(default="oxygen_extractor")

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name="oxygen_extractor",
            map_char="O",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["oxygen"].symbol,
            # Protocols
            max_uses=self.max_uses,
            allow_partial_usage=True,  # can use it while its on cooldown
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    output_resources={"oxygen": 20},
                    cooldown=int(10_000 / self.efficiency),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Rare and doesn't regenerate. But more cogs increase efficiency.
class GermaniumExtractorConfig(ExtractorConfig):
    type: typing.Literal["germanium_extractor"] = pydantic.Field(default="germanium_extractor")
    synergy: int = 1
    efficiency: int = 1

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name="germanium_extractor",
            map_char="G",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["germanium"].symbol,
            # Protocols
            max_uses=self.max_uses,
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(output_resources={"germanium": self.efficiency}),
                *[
                    mettagrid.config.mettagrid_config.ProtocolConfig(
                        vibes=["germanium"] * i, output_resources={"germanium": self.efficiency + i * self.synergy}
                    )
                    for i in range(1, 5)
                ],
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


class SiliconExtractorConfig(ExtractorConfig):
    type: typing.Literal["silicon_extractor"] = pydantic.Field(default="silicon_extractor")

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name="silicon_extractor",
            map_char="S",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["silicon"].symbol,
            # Protocols
            max_uses=max(1, self.max_uses // 10),
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    input_resources={"energy": 25},
                    output_resources={"silicon": max(1, int(25 * self.efficiency // 100))},
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


class CvCChestConfig(CvCStationConfig):
    type: typing.Literal["communal_chest"] = pydantic.Field(default="communal_chest")
    initial_inventory: dict[str, int] = pydantic.Field(
        default={}, description="Initial inventory for each resource type"
    )

    def station_cfg(self) -> mettagrid.config.mettagrid_config.ChestConfig:
        return mettagrid.config.mettagrid_config.ChestConfig(
            name=self.type,
            map_char="C",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["chest"].symbol,
            vibe_transfers={
                "default": {"heart": 255, "carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255},
                "heart": {"heart": -1},
                "carbon": {"carbon": -10},
                "oxygen": {"oxygen": -10},
                "germanium": {"germanium": -1},
                "silicon": {"silicon": -25},
            },
            initial_inventory=self.initial_inventory,
        )


class CvCAssemblerConfig(CvCStationConfig):
    type: typing.Literal["assembler"] = pydantic.Field(default="assembler")
    heart_cost: int = pydantic.Field(default=10)

    def station_cfg(self) -> mettagrid.config.mettagrid_config.AssemblerConfig:
        gear = [("oxygen", "modulator"), ("germanium", "scrambler"), ("silicon", "resonator"), ("carbon", "decoder")]
        return mettagrid.config.mettagrid_config.AssemblerConfig(
            name=self.type,
            map_char="&",
            render_symbol=cogames.cogs_vs_clips.vibes.VIBE_BY_NAME["assembler"].symbol,
            clip_immune=True,
            protocols=[
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    vibes=["heart"] * (i + 1),
                    input_resources={
                        "carbon": self.heart_cost * 2,
                        "oxygen": self.heart_cost * 2,
                        "germanium": max(self.heart_cost // 2 - i, 1),
                        "silicon": self.heart_cost * 5,
                        "energy": self.heart_cost * 2,
                    },
                    output_resources={"heart": 1},
                )
                for i in range(4)
            ]
            + [
                mettagrid.config.mettagrid_config.ProtocolConfig(
                    vibes=["gear", gear[i][0]],
                    input_resources={gear[i][0]: 1},
                    output_resources={gear[i][1]: 1},
                )
                for i in range(len(gear))
            ],
        )
