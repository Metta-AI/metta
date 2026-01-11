from __future__ import annotations

from abc import ABC
from typing import override

from pydantic import Field

from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.stations import (
    COGSGUARD_ELEMENTS,
    COGSGUARD_GEAR,
    CarbonExtractorConfig,
    ChargerConfig,
    CogsGuardChestConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GearStationConfig,
    GermaniumExtractorConfig,
    HubConfig,
    JunctionConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
    SimpleExtractorConfig,
    resources,
)
from mettagrid.base_config import Config
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeVibeActionConfig,
    ClipperConfig,
    CollectiveConfig,
    GameConfig,
    GlobalObsConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ProtocolConfig,
    ResourceLimitsConfig,
    TransferActionConfig,
    VibeTransfer,
)
from mettagrid.config.vibes import Vibe
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

    def compat(self, mission: Mission) -> bool:
        """Check if this variant is compatible with the given mission.

        Returns True if the variant can be safely applied to the mission.
        Override this method to add compatibility checks.
        """
        return True

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

    # Variants are applied to the mission immediately, and to its env when make_env is called
    variants: list[MissionVariant] = Field(default_factory=list)

    carbon_extractor: CarbonExtractorConfig = Field(default_factory=CarbonExtractorConfig)
    oxygen_extractor: OxygenExtractorConfig = Field(default_factory=OxygenExtractorConfig)
    germanium_extractor: GermaniumExtractorConfig = Field(default_factory=GermaniumExtractorConfig)
    silicon_extractor: SiliconExtractorConfig = Field(default_factory=SiliconExtractorConfig)
    charger: ChargerConfig = Field(default_factory=ChargerConfig)
    chest: CvCChestConfig = Field(default_factory=CvCChestConfig)
    wall: CvCWallConfig = Field(default_factory=CvCWallConfig)
    assembler: CvCAssemblerConfig = Field(default_factory=CvCAssemblerConfig)

    clip_period: int = Field(default=0)
    cargo_capacity: int = Field(default=100)
    energy_capacity: int = Field(default=100)
    energy_regen_amount: int = Field(default=1)
    inventory_regen_interval: int = Field(default=1)
    gear_capacity: int = Field(default=5)
    move_energy_cost: int = Field(default=2)
    heart_capacity: int = Field(default=1)
    # Control vibe swapping in variants
    enable_vibe_change: bool = Field(default=True)
    vibes: list[Vibe] | None = Field(default=None)
    compass_enabled: bool = Field(default=True)

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

        game = GameConfig(
            map_builder=map_builder,
            num_agents=num_cogs,
            resource_names=resources,
            vibe_names=[vibe.name for vibe in vibes.VIBES],
            global_obs=GlobalObsConfig(compass=self.compass_enabled, goal_obs=True),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources={"energy": self.move_energy_cost}),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(
                    vibes=(
                        []
                        if not self.enable_vibe_change
                        else (self.vibes if self.vibes is not None else list(vibes.VIBES))
                    )
                ),
                transfer=TransferActionConfig(
                    enabled=True,
                    vibe_transfers=[VibeTransfer(vibe="charger", target={"energy": 20}, actor={"energy": -20})],
                ),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(
                    limits={
                        "heart": ResourceLimitsConfig(limit=self.heart_capacity, resources=["heart"]),
                        "energy": ResourceLimitsConfig(limit=self.energy_capacity, resources=["energy"]),
                        "cargo": ResourceLimitsConfig(
                            limit=self.cargo_capacity, resources=["carbon", "oxygen", "germanium", "silicon"]
                        ),
                        "gear": ResourceLimitsConfig(
                            limit=self.gear_capacity, resources=["scrambler", "modulator", "decoder", "resonator"]
                        ),
                    },
                    initial={"energy": self.energy_capacity},
                    regen_amounts={"default": {"energy": self.energy_regen_amount}},
                ),
                rewards=AgentRewards(
                    # Reward only the agent that deposits a heart.
                    stats={"chest.heart.deposited_by_agent": 1.0},
                ),
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
                clip_period=self.clip_period,
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
                # Resource-specific chests used by diagnostic missions
                # These use simplified vibe_transfers (only "default") to avoid issues when vibes are restricted
                "chest_carbon": self.chest.station_cfg().model_copy(
                    update={
                        "map_name": "chest_carbon",
                        "vibe_transfers": {"default": {"carbon": 255}},
                    }
                ),
                "chest_oxygen": self.chest.station_cfg().model_copy(
                    update={
                        "map_name": "chest_oxygen",
                        "vibe_transfers": {"default": {"oxygen": 255}},
                    }
                ),
                "chest_germanium": self.chest.station_cfg().model_copy(
                    update={
                        "map_name": "chest_germanium",
                        "vibe_transfers": {"default": {"germanium": 255}},
                    }
                ),
                "chest_silicon": self.chest.station_cfg().model_copy(
                    update={
                        "map_name": "chest_silicon",
                        "vibe_transfers": {"default": {"silicon": 255}},
                    }
                ),
                # Clipped variants with unique map_names so they don't conflict with regular extractors
                # These are used by maps that explicitly place clipped extractors
                "clipped_carbon_extractor": self.carbon_extractor.model_copy(update={"start_clipped": True})
                .station_cfg()
                .model_copy(update={"map_name": "clipped_carbon_extractor"}),
                "clipped_oxygen_extractor": self.oxygen_extractor.model_copy(update={"start_clipped": True})
                .station_cfg()
                .model_copy(update={"map_name": "clipped_oxygen_extractor"}),
                "clipped_germanium_extractor": self.germanium_extractor.model_copy(update={"start_clipped": True})
                .station_cfg()
                .model_copy(update={"map_name": "clipped_germanium_extractor"}),
                "clipped_silicon_extractor": self.silicon_extractor.model_copy(update={"start_clipped": True})
                .station_cfg()
                .model_copy(update={"map_name": "clipped_silicon_extractor"}),
            },
        )

        env = MettaGridConfig(game=game)
        # Precaution - copy the env, in case the code above uses some global variable that we don't want to modify.
        # This allows variants to modify the env without copying it again.
        env = env.model_copy(deep=True)
        env.label = self.full_name()

        for variant in self.variants:
            variant.modify_env(self, env)
            env.label += f".{variant.name}"

        return env


# CogsGuard vibes
COGSGUARD_VIBES = [
    Vibe("😐", "default"),
    Vibe("❤️", "heart"),
    Vibe("⚙️", "gear"),
    Vibe("🌀", "scrambler"),
    Vibe("🔗", "aligner"),
    Vibe("⛏️", "miner"),
    Vibe("🔭", "scout"),
]


class CogsGuardMission(Config):
    """Mission configuration for CogsGuard game mode."""

    name: str
    description: str
    site: Site
    num_cogs: int | None = None

    # Game parameters
    max_steps: int = Field(default=1000)
    inventory_regen_interval: int = Field(default=1)

    # Agent configuration
    cog: CogConfig = Field(default_factory=CogConfig)

    # Collective initial resources
    collective_initial_carbon: int = Field(default=10)
    collective_initial_oxygen: int = Field(default=10)
    collective_initial_germanium: int = Field(default=10)
    collective_initial_silicon: int = Field(default=10)
    collective_initial_heart: int = Field(default=5)

    # Station configs
    wall: CvCWallConfig = Field(default_factory=CvCWallConfig)

    def full_name(self) -> str:
        return f"{self.site.name}{MAP_MISSION_DELIMITER}{self.name}"

    def make_env(self) -> MettaGridConfig:
        """Create a MettaGridConfig from this mission."""
        map_builder = self.site.map_builder
        num_cogs = self.num_cogs if self.num_cogs is not None else self.site.min_cogs

        gear = COGSGUARD_GEAR
        elements = COGSGUARD_ELEMENTS
        resources_list = ["energy", "heart", "hp", "influence", *elements, *gear]
        vibe_names = [vibe.name for vibe in COGSGUARD_VIBES]

        extractor_objects = {
            f"{resource}_extractor": SimpleExtractorConfig(resource=resource).station_cfg() for resource in elements
        }
        gear_objects = {f"{g}_station": GearStationConfig(gear_type=g).station_cfg() for g in gear}

        game = GameConfig(
            map_builder=map_builder,
            max_steps=self.max_steps,
            num_agents=num_cogs,
            resource_names=resources_list,
            vibe_names=vibe_names,
            global_obs=GlobalObsConfig(),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources={"energy": self.cog.move_energy_cost}),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(vibes=COGSGUARD_VIBES),
            ),
            agent=AgentConfig(
                collective="cogs",
                inventory=InventoryConfig(
                    limits={
                        "gear": ResourceLimitsConfig(limit=self.cog.gear_limit, resources=gear),
                        "hp": ResourceLimitsConfig(
                            limit=self.cog.hp_limit, resources=["hp"], modifiers=self.cog.hp_modifiers
                        ),
                        "heart": ResourceLimitsConfig(limit=self.cog.heart_limit, resources=["heart"]),
                        "energy": ResourceLimitsConfig(
                            limit=self.cog.energy_limit, resources=["energy"], modifiers=self.cog.energy_modifiers
                        ),
                        "cargo": ResourceLimitsConfig(
                            limit=self.cog.cargo_limit, resources=elements, modifiers=self.cog.cargo_modifiers
                        ),
                        "influence": ResourceLimitsConfig(
                            limit=self.cog.influence_limit,
                            resources=["influence"],
                            modifiers=self.cog.influence_modifiers,
                        ),
                    },
                    initial={"energy": self.cog.initial_energy, "hp": self.cog.initial_hp},
                    regen_amounts={
                        "default": {
                            "energy": self.cog.energy_regen,
                            "hp": self.cog.hp_regen,
                            "influence": self.cog.influence_regen,
                        },
                    },
                ),
                rewards=AgentRewards(
                    collective_stats={
                        "aligned.junction.held": 1.0 / self.max_steps,
                    },
                ),
            ),
            inventory_regen_interval=self.inventory_regen_interval,
            objects={
                "wall": self.wall.station_cfg(),
                "assembler": HubConfig(map_name="assembler", team="cogs").station_cfg(),
                "junction": JunctionConfig(map_name="charger", team="clips").station_cfg(),
                "chest": CogsGuardChestConfig().station_cfg(),
                **extractor_objects,
                **gear_objects,
            },
            collectives={
                "cogs": CollectiveConfig(
                    inventory=InventoryConfig(
                        limits={
                            "resources": ResourceLimitsConfig(limit=10000, resources=elements),
                            "hearts": ResourceLimitsConfig(limit=65535, resources=["heart"]),
                        },
                        initial={
                            "carbon": self.collective_initial_carbon,
                            "oxygen": self.collective_initial_oxygen,
                            "germanium": self.collective_initial_germanium,
                            "silicon": self.collective_initial_silicon,
                            "heart": self.collective_initial_heart,
                        },
                    ),
                ),
                "clips": CollectiveConfig(),
            },
        )

        env = MettaGridConfig(game=game)
        env = env.model_copy(deep=True)
        env.label = self.full_name()

        return env
