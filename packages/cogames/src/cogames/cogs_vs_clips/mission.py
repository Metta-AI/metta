from __future__ import annotations

from abc import ABC
from typing import override

from pydantic import Field

from cogames.cogs_vs_clips import vibes
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCStationConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
    resources,
)
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeVibeActionConfig,
    ClipperConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ProtocolConfig,
)
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig


class MissionVariant(Config, ABC):
    # Note: we could derive the name from the class name automatically, but it would make it
    # harder to find the variant source code based on CLI interactions.
    name: str
    description: str = Field(default="")

    def modify_mission(self, mission: Mission) -> None:
        # Override this method to modify the mission.
        # Variants are allowed to modify the mission in-place - it's guaranteed to be a one-time only instance.
        pass

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        # Override this method to modify the produced environment.
        # Variants are allowed to modify the environment in-place.
        pass

    def apply(self, mission: Mission) -> Mission:
        mission = mission.model_copy(deep=True)
        mission.variants.append(self)
        self.modify_mission(mission)
        return mission

    # Temporary helper useful as long as we have one-time variants in missions.py file.
    def as_mission(self, name: str, description: str, site: Site) -> Mission:
        return Mission(
            name=name,
            description=description,
            site=site,
            variants=[self],
        )


class NumCogsVariant(MissionVariant):
    name: str = "num_cogs"
    description: str = "Set the number of cogs for the mission."
    num_cogs: int

    @override
    def modify_mission(self, mission: Mission) -> None:
        if self.num_cogs < mission.site.min_cogs or self.num_cogs > mission.site.max_cogs:
            raise ValueError(
                f"Invalid number of cogs for {mission.site.name}: {self.num_cogs}. "
                + f"Must be between {mission.site.min_cogs} and {mission.site.max_cogs}"
            )

        mission.num_cogs = self.num_cogs


class Site(Config):
    name: str
    description: str
    map_builder: AnyMapBuilderConfig

    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


MAP_MISSION_DELIMITER = "."


class Mission(Config):
    """Mission configuration for Cogs vs Clips."""

    name: str
    description: str
    site: Site
    num_cogs: int | None = None

    carbon_extractor: CarbonExtractorConfig = Field(default_factory=CarbonExtractorConfig)
    oxygen_extractor: OxygenExtractorConfig = Field(default_factory=OxygenExtractorConfig)
    germanium_extractor: GermaniumExtractorConfig = Field(default_factory=GermaniumExtractorConfig)
    silicon_extractor: SiliconExtractorConfig = Field(default_factory=SiliconExtractorConfig)
    charger: ChargerConfig = Field(default_factory=ChargerConfig)
    chest: CvCChestConfig = Field(default_factory=CvCChestConfig)
    wall: CvCWallConfig = Field(default_factory=CvCWallConfig)
    assembler: CvCAssemblerConfig = Field(default_factory=CvCAssemblerConfig)

    clip_rate: float = Field(default=0.0)
    cargo_capacity: int = Field(default=255)
    energy_capacity: int = Field(default=100)
    energy_regen_amount: int = Field(default=1)
    inventory_regen_interval: int = Field(default=1)
    gear_capacity: int = Field(default=5)
    move_energy_cost: int = Field(default=2)
    heart_capacity: int = Field(default=1)
    # Control vibe swapping in variants
    enable_vibe_change: bool = Field(default=True)
    vibe_count: int | None = Field(default=None)
    compass_enabled: bool = Field(default=False)

    # Variants are applied to the mission immediately, and to its env when make_env is called
    variants: list[MissionVariant] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Can't call `variant.apply` here because it will create a new mission instance
        for variant in self.variants:
            variant.modify_mission(self)

    def with_variants(self, variants: list[MissionVariant]) -> Mission:
        mission = self
        for variant in variants:
            mission = variant.apply(mission)
        return mission

    def full_name(self) -> str:
        return f"{self.site.name}{MAP_MISSION_DELIMITER}{self.name}"

    def make_env(self) -> MettaGridConfig:
        """Create a MettaGridConfig from this mission.

        Applies all variants to the produced configuration.

        Returns:
            MettaGridConfig ready for environment creation
        """
        map_builder = self.site.map_builder
        num_cogs = self.num_cogs if self.num_cogs is not None else self.site.min_cogs

        def _clipped_station_cfg(config: CvCStationConfig, clipped_name: str):
            """Clone a station config with unique names for clipped variants."""
            clipped_cfg = config.model_copy(update={"start_clipped": True})
            station = clipped_cfg.station_cfg()
            station.name = clipped_name
            station.map_name = clipped_name
            return station

        game = GameConfig(
            map_builder=map_builder,
            num_agents=num_cogs,
            resource_names=resources,
            vibe_names=[vibe.name for vibe in vibes.VIBES],
            global_obs=GlobalObsConfig(compass=self.compass_enabled),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources={"energy": self.move_energy_cost}),
                noop=NoopActionConfig(),
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
                    stats={"chest.heart.amount": 1 / num_cogs},
                ),
                initial_inventory={
                    "energy": self.energy_capacity,
                },
                shareable_resources=["energy"],
                inventory_regen_amounts={"energy": self.energy_regen_amount},
                diversity_tracked_resources=["energy", "carbon", "oxygen", "germanium", "silicon", "heart"],
            ),
            inventory_regen_interval=self.inventory_regen_interval,
            clipper=ClipperConfig(
                unclipping_protocols=[
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
                # Clipped variants
                "clipped_carbon_extractor": _clipped_station_cfg(self.carbon_extractor, "clipped_carbon_extractor"),
                "clipped_oxygen_extractor": _clipped_station_cfg(self.oxygen_extractor, "clipped_oxygen_extractor"),
                "clipped_germanium_extractor": _clipped_station_cfg(
                    self.germanium_extractor, "clipped_germanium_extractor"
                ),
                "clipped_silicon_extractor": _clipped_station_cfg(self.silicon_extractor, "clipped_silicon_extractor"),
            },
        )

        env = MettaGridConfig(game=game)
        # Precaution - copy the env, in case the code above uses some global variable that we don't want to modify.
        # This allows variants to modify the env without copying it again.
        env = env.model_copy(deep=True)

        for variant in self.variants:
            variant.modify_env(self, env)

        return env
