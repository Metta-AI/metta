from typing import Optional

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import (
    ActorCollectiveHas,
    ActorHas,
    Align,
    AOEEffectConfig,
    AssemblerConfig,
    ChestConfig,
    ClearInventoryMutation,
    CollectiveDeposit,
    CollectiveWithdraw,
    GridObjectConfig,
    Handler,
    InventoryConfig,
    ProtocolConfig,
    RemoveAlignment,
    TargetCollectiveUpdate,
    UpdateActor,
    WallConfig,
    Withdraw,
    isAligned,
    isEnemy,
    isNeutral,
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

# CogsGuard constants
COGSGUARD_GEAR = ["aligner", "scrambler", "miner", "scout"]
COGSGUARD_ELEMENTS = ["oxygen", "carbon", "germanium", "silicon"]

COGSGUARD_HEART_COST = {e: 1 for e in COGSGUARD_ELEMENTS}
COGSGUARD_ALIGN_COST = {"heart": 1}
COGSGUARD_SCRAMBLE_COST = {"heart": 1}

COGSGUARD_GEAR_COSTS = {
    "aligner": {"carbon": 3, "oxygen": 1, "germanium": 1, "silicon": 1},
    "scrambler": {"carbon": 1, "oxygen": 3, "germanium": 1, "silicon": 1},
    "miner": {"carbon": 1, "oxygen": 1, "germanium": 3, "silicon": 1},
    "scout": {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 3},
}

COGSGUARD_GEAR_SYMBOLS = {
    "aligner": "🔗",
    "scrambler": "🌀",
    "miner": "⛏️",
    "scout": "🔭",
}


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


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


# ==============================================================================
# CogsGuard Station Configs
# ==============================================================================


class SimpleExtractorConfig(CvCStationConfig):
    """Simple resource extractor with inventory that transfers resources to actors."""

    resource: str = Field(description="The resource to extract")
    initial_amount: int = Field(default=100, description="Initial amount of resource in extractor")
    small_amount: int = Field(default=1, description="Amount extracted without mining equipment")
    large_amount: int = Field(default=10, description="Amount extracted with mining equipment")

    def station_cfg(self) -> ChestConfig:
        return ChestConfig(
            name=f"{self.resource}_extractor",
            map_name=f"{self.resource}_extractor",
            render_symbol="📦",
            on_use_handlers={
                # Order matters: miner first so agents with miner gear get the bonus
                "miner": Handler(
                    filters=[ActorHas({"miner": 1})],
                    mutations=[Withdraw({self.resource: self.large_amount})],
                ),
                "extract": Handler(
                    filters=[],
                    mutations=[Withdraw({self.resource: self.small_amount})],
                ),
            },
            inventory=InventoryConfig(initial={self.resource: self.initial_amount}),
        )


class JunctionConfig(CvCStationConfig):
    """Supply depot that receives element resources via default vibe into collective."""

    map_name: str = Field(description="Map name for this junction")
    team: Optional[str] = Field(default=None, description="Team/collective this junction belongs to")
    aoe_range: int = Field(default=10, description="Range for AOE effects")
    influence_deltas: dict[str, int] = Field(default_factory=lambda: {"influence": 10, "energy": 100, "hp": 100})
    attack_deltas: dict[str, int] = Field(default_factory=lambda: {"hp": -1, "influence": -100})
    elements: list[str] = Field(default_factory=lambda: COGSGUARD_ELEMENTS)
    align_cost: dict[str, int] = Field(default_factory=lambda: COGSGUARD_ALIGN_COST)
    scramble_cost: dict[str, int] = Field(default_factory=lambda: COGSGUARD_SCRAMBLE_COST)

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name="junction",
            map_name=self.map_name,
            render_symbol="📦",
            collective=self.team,
            aoes=[
                AOEEffectConfig(range=self.aoe_range, resource_deltas=self.influence_deltas, filters=[isAligned()]),
                AOEEffectConfig(range=self.aoe_range, resource_deltas=self.attack_deltas, filters=[isEnemy()]),
            ],
            on_use_handlers={
                "deposit": Handler(
                    filters=[isAligned()],
                    mutations=[CollectiveDeposit({resource: 100 for resource in self.elements})],
                ),
                "align": Handler(
                    filters=[isNeutral(), ActorHas({"aligner": 1, "influence": 1, **self.align_cost})],
                    mutations=[UpdateActor(_neg(self.align_cost)), Align()],
                ),
                "scramble": Handler(
                    filters=[isEnemy(), ActorHas({"scrambler": 1, **self.scramble_cost})],
                    mutations=[RemoveAlignment(), UpdateActor(_neg(self.scramble_cost))],
                ),
            },
        )


class HubConfig(JunctionConfig):
    """Main hub with influence AOE effect. A junction without align/scramble handlers."""

    def station_cfg(self) -> GridObjectConfig:
        cfg = super().station_cfg()
        cfg.name = "hub"  # override the name
        del cfg.on_use_handlers["align"]
        del cfg.on_use_handlers["scramble"]
        return cfg


class CogsGuardChestConfig(CvCStationConfig):
    """Chest for heart management in CogsGuard."""

    collective: str = Field(default="cogs", description="Collective this chest belongs to")
    heart_cost: dict[str, int] = Field(default_factory=lambda: COGSGUARD_HEART_COST)

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name="chest",
            map_name="chest",
            render_symbol="📦",
            collective=self.collective,
            on_use_handlers={
                "get_heart": Handler(
                    filters=[isAligned()],
                    mutations=[CollectiveWithdraw({"heart": 1})],
                ),
                "make_heart": Handler(
                    filters=[isAligned(), ActorCollectiveHas(self.heart_cost)],
                    mutations=[
                        TargetCollectiveUpdate(_neg(self.heart_cost)),
                        UpdateActor({"heart": 1}),
                    ],
                ),
            },
        )


class GearStationConfig(CvCStationConfig):
    """Gear station that clears all gear and adds the specified gear type."""

    gear_type: str = Field(description="Type of gear this station provides")
    collective: str = Field(default="cogs", description="Collective this station belongs to")
    gear_costs: dict[str, dict[str, int]] = Field(default_factory=lambda: COGSGUARD_GEAR_COSTS)

    def station_cfg(self) -> GridObjectConfig:
        cost = self.gear_costs.get(self.gear_type, {})
        return GridObjectConfig(
            name=f"{self.gear_type}_station",
            map_name=f"{self.gear_type}_station",
            render_symbol=COGSGUARD_GEAR_SYMBOLS.get(self.gear_type, "⚙️"),
            collective=self.collective,
            on_use_handlers={
                "keep_gear": Handler(
                    filters=[isAligned(), ActorHas({self.gear_type: 1})],
                    mutations=[],
                ),
                "change_gear": Handler(
                    filters=[isAligned(), ActorCollectiveHas(cost)],
                    mutations=[
                        ClearInventoryMutation(target="actor", limit_name="gear"),
                        TargetCollectiveUpdate(_neg(cost)),
                        UpdateActor({self.gear_type: 1}),
                    ],
                ),
            },
        )
