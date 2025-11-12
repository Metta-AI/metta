from pydantic import Field

from cogames.cogs_vs_clips import vibes
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
    def station_cfg(self) -> WallConfig:
        return WallConfig(name="wall", map_char="#", render_symbol=vibes.VIBE_BY_NAME["wall"].symbol)


class ExtractorConfig(CvCStationConfig):
    """Base class for all extractor configs."""

    max_uses: int = Field(default=1000)
    efficiency: int = Field(default=100)


class ChargerConfig(ExtractorConfig):
    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="charger",
            map_char="+",
            render_symbol=vibes.VIBE_BY_NAME["charger"].symbol,
            # Protocols
            allow_partial_usage=True,  # can use it while its on cooldown
            max_uses=0,  # unlimited uses
            protocols=[
                ProtocolConfig(
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
    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="carbon_extractor",
            map_char="C",
            render_symbol=vibes.VIBE_BY_NAME["carbon"].symbol,
            # Protocols
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
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
    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="oxygen_extractor",
            map_char="O",
            render_symbol=vibes.VIBE_BY_NAME["oxygen"].symbol,
            # Protocols
            max_uses=self.max_uses,
            allow_partial_usage=True,  # can use it while its on cooldown
            protocols=[
                ProtocolConfig(
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
    synergy: int = 1
    efficiency: int = 1

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="germanium_extractor",
            map_char="G",
            render_symbol=vibes.VIBE_BY_NAME["germanium"].symbol,
            # Protocols
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(output_resources={"germanium": self.efficiency}),
                *[
                    ProtocolConfig(
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
    max_uses: int = Field(default=100)  # Silicon has lower default than other extractors

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="silicon_extractor",
            map_char="S",
            render_symbol=vibes.VIBE_BY_NAME["silicon"].symbol,
            # Protocols
            max_uses=self.max_uses,  # Use direct value, no division
            protocols=[
                ProtocolConfig(
                    input_resources={"energy": 25},
                    output_resources={"silicon": max(1, int(25 * self.efficiency // 100))},
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


class CvCChestConfig(CvCStationConfig):
    initial_inventory: dict[str, int] = Field(default={}, description="Initial inventory for each resource type")

    def station_cfg(self) -> ChestConfig:
        # Use map_name/name "chest" so maps and procedural builders that place
        # "chest" resolve to this config. The specific CvC type remains a label.
        return ChestConfig(
            map_char="C",
            render_symbol=vibes.VIBE_BY_NAME["chest"].symbol,
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
    # These could be "fixed_cost" and "variable_cost" instead, but we're more likely to want to read them like this.
    first_heart_cost: int = Field(default=10)
    additional_heart_cost: int = Field(default=5)

    def station_cfg(self) -> AssemblerConfig:
        gear = [("oxygen", "modulator"), ("germanium", "scrambler"), ("silicon", "resonator"), ("carbon", "decoder")]
        return AssemblerConfig(
            name="assembler",
            map_char="&",
            render_symbol=vibes.VIBE_BY_NAME["assembler"].symbol,
            clip_immune=True,
            protocols=[
                ProtocolConfig(
                    vibes=["heart"] * (i + 1),
                    input_resources={
                        "carbon": 2 * (self.first_heart_cost + self.additional_heart_cost * i),
                        "oxygen": 2 * (self.first_heart_cost + self.additional_heart_cost * i),
                        "germanium": max(1, (self.first_heart_cost + self.additional_heart_cost * i) // 2),
                        "silicon": 5 * (self.first_heart_cost + self.additional_heart_cost * i),
                        "energy": 2 * (self.first_heart_cost + self.additional_heart_cost * i),
                    },
                    output_resources={"heart": i + 1},
                )
                for i in range(4)
            ]
            + [
                # Specific gear protocols: ['gear', 'resource'] -> gear_item
                # Agent must have the specific resource AND use gear vibe
                ProtocolConfig(
                    vibes=["gear", gear[i][0]],
                    input_resources={gear[i][0]: 1},
                    output_resources={gear[i][1]: 1},
                )
                for i in range(len(gear))
            ],
            # Note: Generic ['gear'] protocol is added dynamically by clipping variants
            # C++ only allows ONE protocol per unique vibe list, so we can't pre-add all 4 here
        )
