from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config import vibes
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
        return WallConfig(name="wall", render_symbol=vibes.VIBE_BY_NAME["wall"].symbol)


class ExtractorConfig(CvCStationConfig):
    """Base class for all extractor configs."""

    efficiency: int = Field(default=100)


class ChargerConfig(ExtractorConfig):
    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="charger",
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
    max_uses: int = Field(default=25)

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="carbon_extractor",
            render_symbol=vibes.VIBE_BY_NAME["carbon_a"].symbol,
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    output_resources={"carbon": 2 * self.efficiency // 100},
                    cooldown=0,
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Accumulates over time.
class OxygenExtractorConfig(ExtractorConfig):
    max_uses: int = Field(default=5)

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="oxygen_extractor",
            render_symbol=vibes.VIBE_BY_NAME["oxygen_a"].symbol,
            max_uses=self.max_uses,
            allow_partial_usage=True,  # can use it while its on cooldown
            protocols=[
                ProtocolConfig(
                    output_resources={"oxygen": 10},
                    cooldown=int(10_000 / self.efficiency),
                )
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Rare and doesn't regenerate. But more cogs increase efficiency.
class GermaniumExtractorConfig(ExtractorConfig):
    # How much one agent gets.
    efficiency: int = 2
    # How much each additional agent gets.
    synergy: int = 1

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="germanium_extractor",
            render_symbol=vibes.VIBE_BY_NAME["germanium_a"].symbol,
            # Germanium is inherently a single use resource.
            max_uses=1,
            protocols=[
                ProtocolConfig(
                    # For the 1 agent protocol, we set min_agents to zero so it's visible when no
                    # agents are adjacent to the extractor.
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    output_resources={"germanium": self.efficiency + additional_agents * self.synergy},
                )
                for additional_agents in range(4)
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Bulky and energy intensive.
class SiliconExtractorConfig(ExtractorConfig):
    max_uses: int = Field(default=10)

    def station_cfg(self) -> AssemblerConfig:
        return AssemblerConfig(
            name="silicon_extractor",
            render_symbol=vibes.VIBE_BY_NAME["silicon_a"].symbol,
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    input_resources={"energy": 20},
                    output_resources={"silicon": max(1, int(15 * self.efficiency // 100))},
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
            render_symbol=vibes.VIBE_BY_NAME["chest"].symbol,
            vibe_transfers={
                "default": {"heart": 255, "carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255},
                "heart_a": {"heart": -1},
                "heart_b": {"heart": 1},
                "carbon_a": {"carbon": -10},
                "carbon_b": {"carbon": 10},
                "oxygen_a": {"oxygen": -10},
                "oxygen_b": {"oxygen": 10},
                "germanium_a": {"germanium": -1},
                "germanium_b": {"germanium": 1},
                "silicon_a": {"silicon": -25},
                "silicon_b": {"silicon": 25},
            },
            initial_inventory=self.initial_inventory,
        )


class CvCAssemblerConfig(CvCStationConfig):
    # These could be "fixed_cost" and "variable_cost" instead, but we're more likely to want to read them like this.
    first_heart_cost: int = Field(default=10)
    additional_heart_cost: int = Field(default=5)

    def station_cfg(self) -> AssemblerConfig:
        gear = [("carbon", "decoder"), ("oxygen", "modulator"), ("germanium", "scrambler"), ("silicon", "resonator")]
        return AssemblerConfig(
            name="assembler",
            render_symbol=vibes.VIBE_BY_NAME["assembler"].symbol,
            clip_immune=True,
            protocols=[
                ProtocolConfig(
                    vibes=["heart_a"] * (i + 1),
                    input_resources={
                        "carbon": self.first_heart_cost + self.additional_heart_cost * i,
                        "oxygen": self.first_heart_cost + self.additional_heart_cost * i,
                        "germanium": max(1, (self.first_heart_cost + self.additional_heart_cost * i) // 5),
                        "silicon": 3 * (self.first_heart_cost + self.additional_heart_cost * i),
                    },
                    output_resources={"heart": i + 1},
                )
                for i in range(4)
            ]
            + [
                # Specific gear protocols: ['gear', 'resource'] -> gear_item
                # Agent must have the specific resource AND use gear vibe
                ProtocolConfig(
                    vibes=["gear", f"{gear[i][0]}_a"],
                    input_resources={gear[i][0]: 1},
                    output_resources={gear[i][1]: 1},
                )
                for i in range(len(gear))
            ],
            # Note: Generic ['gear'] protocol is added dynamically by clipping variants
            # C++ only allows ONE protocol per unique vibe list, so we can't pre-add all 4 here
        )
