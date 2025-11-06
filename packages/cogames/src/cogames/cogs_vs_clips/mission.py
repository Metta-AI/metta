
import abc
import typing

import pydantic

import cogames.cogs_vs_clips
import cogames.cogs_vs_clips.stations
import mettagrid.base_config
import mettagrid.config.mettagrid_config
import mettagrid.map_builder.map_builder


class MissionVariant(mettagrid.base_config.Config, abc.ABC):
    # Note: we could derive the name from the class name automatically, but it would make it
    # harder to find the variant source code based on CLI interactions.
    name: str
    description: str = pydantic.Field(default="")

    def modify_mission(self, mission: Mission) -> None:
        # Override this method to modify the mission.
        # Variants are allowed to modify the mission in-place - it's guaranteed to be a one-time only instance.
        pass

    def modify_env(self, mission: Mission, env: mettagrid.config.mettagrid_config.MettaGridConfig) -> None:
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

    @typing.override
    def modify_mission(self, mission: Mission) -> None:
        if self.num_cogs < mission.site.min_cogs or self.num_cogs > mission.site.max_cogs:
            raise ValueError(
                f"Invalid number of cogs for {mission.site.name}: {self.num_cogs}. "
                + f"Must be between {mission.site.min_cogs} and {mission.site.max_cogs}"
            )

        mission.num_cogs = self.num_cogs


class Site(mettagrid.base_config.Config):
    name: str
    description: str
    map_builder: mettagrid.map_builder.map_builder.AnyMapBuilderConfig

    min_cogs: int = pydantic.Field(default=1, ge=1)
    max_cogs: int = pydantic.Field(default=1000, ge=1)


MAP_MISSION_DELIMITER = "."


class Mission(mettagrid.base_config.Config):
    """Mission configuration for Cogs vs Clips."""

    name: str
    description: str
    site: Site
    num_cogs: int | None = None

    carbon_extractor: cogames.cogs_vs_clips.stations.CarbonExtractorConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.CarbonExtractorConfig
    )
    oxygen_extractor: cogames.cogs_vs_clips.stations.OxygenExtractorConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.OxygenExtractorConfig
    )
    germanium_extractor: cogames.cogs_vs_clips.stations.GermaniumExtractorConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.GermaniumExtractorConfig
    )
    silicon_extractor: cogames.cogs_vs_clips.stations.SiliconExtractorConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.SiliconExtractorConfig
    )
    charger: cogames.cogs_vs_clips.stations.ChargerConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.ChargerConfig
    )
    chest: cogames.cogs_vs_clips.stations.CvCChestConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.CvCChestConfig
    )
    wall: cogames.cogs_vs_clips.stations.CvCWallConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.CvCWallConfig
    )
    assembler: cogames.cogs_vs_clips.stations.CvCAssemblerConfig = pydantic.Field(
        default_factory=cogames.cogs_vs_clips.stations.CvCAssemblerConfig
    )

    clip_rate: float = pydantic.Field(default=0.0)
    cargo_capacity: int = pydantic.Field(default=255)
    energy_capacity: int = pydantic.Field(default=100)
    energy_regen_amount: int = pydantic.Field(default=1)
    inventory_regen_interval: int = pydantic.Field(default=1)
    gear_capacity: int = pydantic.Field(default=5)
    move_energy_cost: int = pydantic.Field(default=2)
    heart_capacity: int = pydantic.Field(default=1)
    # Control vibe swapping in variants
    enable_vibe_change: bool = pydantic.Field(default=True)
    vibe_count: int | None = pydantic.Field(default=None)

    # Variants are applied to the mission immediately, and to its env when make_env is called
    variants: list[MissionVariant] = pydantic.Field(default_factory=list)

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

    def make_env(self) -> mettagrid.config.mettagrid_config.MettaGridConfig:
        """Create a MettaGridConfig from this mission.

        Applies all variants to the produced configuration.

        Returns:
            MettaGridConfig ready for environment creation
        """
        map_builder = self.site.map_builder
        num_cogs = self.num_cogs if self.num_cogs is not None else self.site.min_cogs

        game = mettagrid.config.mettagrid_config.GameConfig(
            map_builder=map_builder,
            num_agents=num_cogs,
            resource_names=cogames.cogs_vs_clips.stations.resources,
            vibe_names=[vibe.name for vibe in cogames.cogs_vs_clips.vibes.VIBES],
            actions=mettagrid.config.mettagrid_config.ActionsConfig(
                move=mettagrid.config.mettagrid_config.MoveActionConfig(
                    consumed_resources={"energy": self.move_energy_cost}
                ),
                noop=mettagrid.config.mettagrid_config.NoopActionConfig(),
                change_vibe=mettagrid.config.mettagrid_config.ChangeVibeActionConfig(
                    number_of_vibes=(
                        0
                        if not self.enable_vibe_change
                        else (
                            self.vibe_count if self.vibe_count is not None else len(cogames.cogs_vs_clips.vibes.VIBES)
                        )
                    )
                ),
            ),
            agent=mettagrid.config.mettagrid_config.AgentConfig(
                resource_limits={
                    "heart": self.heart_capacity,
                    "energy": self.energy_capacity,
                    ("carbon", "oxygen", "germanium", "silicon"): self.cargo_capacity,
                    ("scrambler", "modulator", "decoder", "resonator"): self.gear_capacity,
                },
                rewards=mettagrid.config.mettagrid_config.AgentRewards(
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
            clipper=mettagrid.config.mettagrid_config.ClipperConfig(
                unclipping_protocols=[
                    mettagrid.config.mettagrid_config.ProtocolConfig(
                        input_resources={"decoder": 1},
                        cooldown=1,
                    ),
                    mettagrid.config.mettagrid_config.ProtocolConfig(
                        input_resources={"modulator": 1},
                        cooldown=1,
                    ),
                    mettagrid.config.mettagrid_config.ProtocolConfig(
                        input_resources={"scrambler": 1},
                        cooldown=1,
                    ),
                    mettagrid.config.mettagrid_config.ProtocolConfig(
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
                "clipped_carbon_extractor": self.carbon_extractor.model_copy(
                    update={"start_clipped": True}
                ).station_cfg(),
                "clipped_oxygen_extractor": self.oxygen_extractor.model_copy(
                    update={"start_clipped": True}
                ).station_cfg(),
                "clipped_germanium_extractor": self.germanium_extractor.model_copy(
                    update={"start_clipped": True}
                ).station_cfg(),
                "clipped_silicon_extractor": self.silicon_extractor.model_copy(
                    update={"start_clipped": True}
                ).station_cfg(),
            },
        )

        env = mettagrid.config.mettagrid_config.MettaGridConfig(game=game)
        # Precaution - copy the env, in case the code above uses some global variable that we don't want to modify.
        # This allows variants to modify the env without copying it again.
        env = env.model_copy(deep=True)

        for variant in self.variants:
            variant.modify_env(self, env)

        return env
