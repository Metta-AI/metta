from typing import Literal

from pydantic import Field

from cogames.cogs_vs_clips import vibes
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
    resources,
)
from mettagrid.base_config import Config
from mettagrid.builder.envs import ActionConfig
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    ClipperConfig,
    GameConfig,
    MettaGridConfig,
    ProtocolConfig,
)
from mettagrid.map_builder.map_builder import MapBuilderConfig


class MissionVariant(Config):
    name: str = Field()
    description: str = Field()

    def apply(self, mission: "Mission") -> "Mission":
        return mission


class Site(Config):
    name: str
    description: str
    map_builder: MapBuilderConfig
    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


class Mission(Config):
    """Mission configuration for Cogs vs Clips.

    This class combines both the mission template (with defaults) and the
    instantiated mission configuration (with specific map and num_cogs).
    """

    name: str = Field()
    description: str = Field()
    site: Site | None = Field(default=None)

    # Map and num_cogs are optional for template missions, required for instantiated missions
    map: MapBuilderConfig | None = Field(default=None)
    num_cogs: int | None = Field(default=None)

    carbon_extractor: CarbonExtractorConfig = Field(default_factory=CarbonExtractorConfig)
    oxygen_extractor: OxygenExtractorConfig = Field(default_factory=OxygenExtractorConfig)
    germanium_extractor: GermaniumExtractorConfig = Field(default_factory=GermaniumExtractorConfig)
    silicon_extractor: SiliconExtractorConfig = Field(default_factory=SiliconExtractorConfig)
    charger: ChargerConfig = Field(default_factory=ChargerConfig)
    chest: CvCChestConfig = Field(default_factory=CvCChestConfig)
    wall: CvCWallConfig = Field(default_factory=CvCWallConfig)
    assembler: CvCAssemblerConfig = Field(default_factory=CvCAssemblerConfig)

    clip_rate: float = Field(default=0.0)
    cargo_capacity: int = Field(default=100)
    energy_capacity: int = Field(default=100)
    energy_regen_amount: int = Field(default=1)
    gear_capacity: int = Field(default=5)
    move_energy_cost: int = Field(default=2)
    heart_capacity: int = Field(default=1)
    change_glyph_enabled: bool = Field(default=True)
    glyph_limit: int | None = Field(default=None, ge=0, le=len(vibes.VIBES))
    reward_profile: Literal["default", "heart_focus"] = Field(default="default")

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        """Create an instantiated mission with specific map and num_cogs.

        Args:
            map_builder: Map configuration
            num_cogs: Number of cogs (agents)
            variant: Optional variant to apply

        Returns:
            New Mission instance with map and num_cogs set
        """
        mission = self.model_copy(deep=True)
        mission.map = map_builder
        mission.num_cogs = num_cogs

        if variant:
            mission = variant.apply(mission)

        return mission

    def make_env(self) -> MettaGridConfig:
        """Create a MettaGridConfig from this mission.

        Requires that map and num_cogs are set (i.e., this is an instantiated mission).

        Returns:
            MettaGridConfig ready for environment creation

        Raises:
            ValueError: If map or num_cogs is not set
        """
        if self.map is None:
            raise ValueError("Cannot make_env without a map. Call instantiate() first.")
        if self.num_cogs is None:
            raise ValueError("Cannot make_env without num_cogs. Call instantiate() first.")

        resource_limits: dict[str | tuple[str, ...], int] = {
            "heart": self.heart_capacity,
            "energy": self.energy_capacity,
            ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
            ("scrambler", "modulator", "decoder", "resonator"): self.gear_capacity,
        }

        reward_stats: dict[str, float]
        if self.reward_profile == "heart_focus":
            reward_stats = {
                "heart.gained": 0.25,
                "chest.heart.deposited": 1.0,
                "carbon.gained": 0.02,
                "oxygen.gained": 0.02,
                "germanium.gained": 0.05,
                "silicon.gained": 0.02,
                "energy.gained": 0.005,
            }
        else:
            reward_stats = {"chest.heart.amount": 1 / self.num_cogs}

        agent_config = AgentConfig(
            resource_limits=resource_limits,
            rewards=AgentRewards(
                stats=reward_stats,
            ),
            initial_inventory={
                "energy": self.energy_capacity,
            },
            shareable_resources=["energy"],
            inventory_regen_amounts={"energy": self.energy_regen_amount},
        )

        game = GameConfig(
            map_builder=self.map,
            num_agents=self.num_cogs,
            resource_names=resources,
            vibe_names=[vibe.name for vibe in vibes.VIBES],
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": self.move_energy_cost}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(
                    enabled=self.change_glyph_enabled,
                    number_of_glyphs=self._resolve_glyph_limit(),
                ),
            ),
            agent=agent_config,
            inventory_regen_interval=1,
            clipper=ClipperConfig(
                unclipping_recipes=[
                    ProtocolConfig(
                        input_resources={"decoder": 1},
                        cooldown=1,
                    ),
                    ProtocolConfig(
                        input_resources={"modulator": 1},
                        cooldown=1,
                    ),
                    ProtocolConfig(
                        input_resources={"scrambler": 1},
                        cooldown=1,
                    ),
                    ProtocolConfig(
                        input_resources={"resonator": 1},
                        cooldown=1,
                    ),
                ],
                clip_rate=self.clip_rate,
            ),
            objects={
                "wall": self.wall.station_cfg(),
                "assembler": self.assembler.station_cfg(),
                "chest": self.chest.station_cfg(),
                "charger": self.charger.station_cfg(),
                "carbon_extractor": self.carbon_extractor.station_cfg(),
                "oxygen_extractor": self.oxygen_extractor.station_cfg(),
                "germanium_extractor": self.germanium_extractor.station_cfg(),
                "silicon_extractor": self.silicon_extractor.station_cfg(),
            },
        )

        return MettaGridConfig(game=game)

    def _resolve_glyph_limit(self) -> int:
        if self.glyph_limit is None:
            return len(vibes.VIBES)
        return max(0, min(self.glyph_limit, len(vibes.VIBES)))
