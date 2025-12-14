from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    GridObjectConfig,
    InventoryConfig,
    ProtocolConfig,
    WallConfig,
)

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

    # How much this extractor produces relative to its default.
    # Efficiency outside of this range won't technically break things, but they'll be far enough from the
    # expectations that we don't want to go beyond them without some thought.
    efficiency: int = Field(ge=20, le=500, default=100)
    # How much additional agents increase production.
    # Scaled so 0 means none and 100 means some version of "twice as much".
    synergy: int = Field(default=0)
    max_uses: int = Field()


class ChargerConfig(ExtractorConfig):
    max_uses: int = 0  # unlimited uses

    def station_cfg(self) -> AssemblerConfig:
        output = 50 * self.efficiency // 100
        return AssemblerConfig(
            name="charger",
            render_symbol=vibes.VIBE_BY_NAME["charger"].symbol,
            # Protocols
            allow_partial_usage=True,  # can use it while its on cooldown
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    output_resources={"energy": output * (100 + additional_agents * self.synergy) // 100},
                    cooldown=10,
                )
                for additional_agents in range(4)
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Time consuming but easy to mine.
class CarbonExtractorConfig(ExtractorConfig):
    max_uses: int = Field(default=25)

    def station_cfg(self) -> AssemblerConfig:
        output = 2 * self.efficiency // 100
        return AssemblerConfig(
            name="carbon_extractor",
            render_symbol=vibes.VIBE_BY_NAME["carbon_a"].symbol,
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    output_resources={"carbon": output * (100 + additional_agents * self.synergy) // 100},
                    cooldown=0,
                )
                for additional_agents in range(4)
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Accumulates over time.
class OxygenExtractorConfig(ExtractorConfig):
    max_uses: int = Field(default=5)

    def station_cfg(self) -> AssemblerConfig:
        # efficiency impacts cooldown, not output
        output = 10
        return AssemblerConfig(
            name="oxygen_extractor",
            render_symbol=vibes.VIBE_BY_NAME["oxygen_a"].symbol,
            max_uses=self.max_uses,
            allow_partial_usage=True,  # can use it while its on cooldown
            protocols=[
                ProtocolConfig(
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    output_resources={"oxygen": output * (100 + additional_agents * self.synergy) // 100},
                    cooldown=int(10_000 / self.efficiency),
                )
                for additional_agents in range(4)
            ],
            # Clipping
            start_clipped=self.start_clipped,
            clip_immune=self.clip_immune,
        )


# Rare regenerates slowly. More cogs increase the amount extracted.
class GermaniumExtractorConfig(ExtractorConfig):
    max_uses: int = Field(default=5)
    synergy: int = 50

    def station_cfg(self) -> AssemblerConfig:
        # efficiency impacts cooldown, not output
        output = 2
        return AssemblerConfig(
            name="germanium_extractor",
            render_symbol=vibes.VIBE_BY_NAME["germanium_a"].symbol,
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    output_resources={"germanium": output * (100 + additional_agents * self.synergy) // 100},
                    cooldown=int(20_000 / self.efficiency),
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
        output = 15 * self.efficiency // 100
        return AssemblerConfig(
            name="silicon_extractor",
            render_symbol=vibes.VIBE_BY_NAME["silicon_a"].symbol,
            max_uses=self.max_uses,
            protocols=[
                ProtocolConfig(
                    min_agents=(additional_agents + 1) if additional_agents >= 1 else 0,
                    input_resources={"energy": 20},
                    output_resources={"silicon": output * (100 + additional_agents * self.synergy) // 100},
                )
                for additional_agents in range(4)
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
                "heart_a": {"heart": 0},
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
            inventory=InventoryConfig(initial=self.initial_inventory),
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
