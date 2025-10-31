from typing import Any

from pydantic import Field

from cogames.cogs_vs_clips import vibes
from cogames.cogs_vs_clips.procedural import apply_procedural_overrides_to_builder
from cogames.cogs_vs_clips.stations import (
    RESOURCE_CHESTS,
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
    ChangeVibeActionConfig,
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
    procedural_overrides: dict[str, Any] = Field(default_factory=dict)

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
    # Control vibe swapping in variants
    enable_vibe_change: bool = Field(default=True)
    vibe_count: int | None = Field(default=None)

    def configure(self):
        pass

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
        *,
        cli_override: bool = False,
    ) -> "Mission":
        """Create an instantiated mission with specific map and num_cogs.

        Args:
            map_builder: Map configuration
            num_cogs: Number of cogs (agents)
            variant: Optional variant to apply
            cli_override: If True, prefer the provided num_cogs over mission/variant settings

        Returns:
            New Mission instance with map and num_cogs set
        """
        mission = self.model_copy(deep=True)
        mission.configure()
        mission.map = map_builder

        if variant:
            mission = variant.apply(mission)

        if cli_override:
            mission.num_cogs = num_cogs
        elif mission.num_cogs is None:
            mission.num_cogs = num_cogs

        # Apply mission-level procedural overrides to supported builders (hub-only, machina, etc.)
        mission.map = apply_procedural_overrides_to_builder(
            mission.map or map_builder,
            num_cogs=int(mission.num_cogs or 0),
            overrides=getattr(mission, "procedural_overrides", {}) or {},
        )

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

        # supervisor_config = PatrolSupervisorConfig(
        #     steps_per_direction=5,
        #     can_override_action=True,
        #     name="patrol_supervisor",
        # )

        game = GameConfig(
            map_builder=self.map,
            num_agents=self.num_cogs,
            resource_names=resources,
            vibe_names=[vibe.name for vibe in vibes.VIBES],
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": self.move_energy_cost}),
                noop=ActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    number_of_vibes=(
                        0
                        if not self.enable_vibe_change
                        else (self.vibe_count if self.vibe_count is not None else len(vibes.VIBES))
                    )
                ),
            ),
            agent=AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                    ("scrambler", "modulator", "decoder", "resonator"): self.gear_capacity,
                },
                rewards=AgentRewards(
                    stats={"chest.heart.amount": 1 / self.num_cogs},
                ),
                initial_inventory={
                    "energy": self.energy_capacity,
                },
                shareable_resources=["energy"],
                inventory_regen_amounts={"energy": self.energy_regen_amount},
                diversity_tracked_resources=["energy", "carbon", "oxygen", "germanium", "silicon"],
                # supervisor=supervisor_config,
            ),
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
                **RESOURCE_CHESTS,
            },
        )

        # if hasattr(self, "heart_chorus_length"):
        #     length_raw = self.heart_chorus_length
        #     try:
        #         chorus_len = max(1, int(length_raw))
        #     except Exception:
        #         chorus_len = 4
        #     inputs = getattr(
        #         self,
        #         "heart_chorus_inputs",
        #         {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1, "energy": 1},
        #     )
        #     assembler_cfg = game.objects.get("assembler")
        #     if isinstance(assembler_cfg, CvCAssemblerConfig):
        #         chorus = ProtocolConfig(input_resources=dict(inputs), output_resources={"heart": 1}, cooldown=1)
        #         non_heart = [
        #             (vibes_list, recipe)
        #             for vibes_list, recipe in assembler_cfg.recipes
        #             if recipe.output_resources.get("heart", 0) == 0
        #         ]
        #         assembler_cfg.recipes = [(["heart"] * chorus_len, chorus), *non_heart]
        return MettaGridConfig(game=game)
