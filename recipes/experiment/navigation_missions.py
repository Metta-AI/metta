"""Navigation missions for CoGames.

These missions wrap the navigation environments defined in recipes.experiment.navigation
so they can be used within the standard Mission/Curriculum infrastructure.
"""

from typing import override

from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from metta.map.terrain_from_numpy import NavigationFromNumpy
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from recipes.experiment import navigation


def _cleanup_nav_env(env: MettaGridConfig) -> MettaGridConfig:
    """Cleanup navigation environment to be compatible with CVC training.

    Ensures the environment configuration satisfies CVC constraints and matches
    the standard action space (using full VIBES) to allow joint training.

    This includes adding all CVC objects and resources even if not used in navigation.
    """
    env.game.vibe_names = [vibe.name for vibe in vibes.VIBES]

    carbon_cfg = CarbonExtractorConfig()
    oxygen_cfg = OxygenExtractorConfig()
    germanium_cfg = GermaniumExtractorConfig()
    silicon_cfg = SiliconExtractorConfig()
    charger_cfg = ChargerConfig()
    chest_cfg = CvCChestConfig()
    wall_cfg = CvCWallConfig()
    assembler_cfg = CvCAssemblerConfig()

    env.game.objects.update(
        {
            "wall": wall_cfg.station_cfg(),
            "assembler": assembler_cfg.station_cfg(),
            "chest": chest_cfg.station_cfg(),
            "charger": charger_cfg.station_cfg(),
            "carbon_extractor": carbon_cfg.station_cfg(),
            "oxygen_extractor": oxygen_cfg.station_cfg(),
            "germanium_extractor": germanium_cfg.station_cfg(),
            "silicon_extractor": silicon_cfg.station_cfg(),
            "chest_carbon": chest_cfg.station_cfg().model_copy(
                update={
                    "map_name": "chest_carbon",
                    "vibe_transfers": {"default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255}},
                }
            ),
            "chest_oxygen": chest_cfg.station_cfg().model_copy(
                update={
                    "map_name": "chest_oxygen",
                    "vibe_transfers": {"default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255}},
                }
            ),
            "chest_germanium": chest_cfg.station_cfg().model_copy(
                update={
                    "map_name": "chest_germanium",
                    "vibe_transfers": {"default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255}},
                }
            ),
            "chest_silicon": chest_cfg.station_cfg().model_copy(
                update={
                    "map_name": "chest_silicon",
                    "vibe_transfers": {"default": {"carbon": 255, "oxygen": 255, "germanium": 255, "silicon": 255}},
                }
            ),
            "clipped_carbon_extractor": carbon_cfg.station_cfg().model_copy(
                update={"map_name": "clipped_carbon_extractor", "start_clipped": True}
            ),
            "clipped_oxygen_extractor": oxygen_cfg.station_cfg().model_copy(
                update={"map_name": "clipped_oxygen_extractor", "start_clipped": True}
            ),
            "clipped_germanium_extractor": germanium_cfg.station_cfg().model_copy(
                update={"map_name": "clipped_germanium_extractor", "start_clipped": True, "max_uses": 1}
            ),
            "clipped_silicon_extractor": silicon_cfg.station_cfg().model_copy(
                update={"map_name": "clipped_silicon_extractor", "start_clipped": True}
            ),
        }
    )

    env.game.resource_names = [
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

    if env.game.actions:
        if env.game.actions.change_vibe:
            env.game.actions.change_vibe.enabled = True
            env.game.actions.change_vibe.vibes = list(vibes.VIBES)

            if env.game.agent.initial_vibe >= len(vibes.VIBES):
                env.game.agent.initial_vibe = 0

        if env.game.actions.attack:
            env.game.actions.attack.enabled = False

    return env


class NavigationMission(Mission):
    """A mission that wraps a navigation environment builder."""

    nav_map_name: str
    max_steps: int = 1000
    num_instances: int = 1

    def __init__(self, name: str, nav_map_name: str, **kwargs):
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name=name,
            description=f"Navigation task on map {nav_map_name}",
            site=HELLO_WORLD,
            nav_map_name=nav_map_name,
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        num_agents = self.num_cogs if self.num_cogs is not None else 1

        env = navigation.make_nav_ascii_env(
            name=self.nav_map_name,
            max_steps=self.max_steps,
            num_agents=num_agents,
            num_instances=self.num_instances,
        )
        return _cleanup_nav_env(env)


class NavigationDenseMission(Mission):
    """A mission that uses the dense varied terrain maps from navigation training."""

    terrain_dir: str

    def __init__(self, name: str, terrain_dir: str, **kwargs):
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name=name,
            description=f"Dense varied terrain navigation: {terrain_dir}",
            site=HELLO_WORLD,
            terrain_dir=terrain_dir,
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        num_agents = self.num_cogs if self.num_cogs is not None else 1

        nav_env = navigation.mettagrid(num_agents=num_agents, num_instances=1)

        map_builder = nav_env.game.map_builder
        assert isinstance(map_builder, MapGen.Config), "Expected MapGen.Config for navigation map builder"

        default_instance = map_builder.instance
        if isinstance(default_instance, NavigationFromNumpy.Config):
            objects = default_instance.objects
        else:
            objects = {"assembler": 10}

        map_builder.instance = NavigationFromNumpy.Config(
            agents=num_agents,
            objects=objects,
            dir=self.terrain_dir,
        )

        return _cleanup_nav_env(nav_env)


NAVIGATION_EVAL_MISSIONS: list[Mission] = [
    NavigationMission(
        name=f"navigation_{eval_cfg['name']}",
        nav_map_name=eval_cfg["name"],
        max_steps=eval_cfg["max_steps"],
        num_cogs=eval_cfg["num_agents"],
        num_instances=1,
    )
    for eval_cfg in navigation.NAVIGATION_EVALS
]

NAVIGATION_MISSIONS: list[Mission] = []

_maps = []
for size in ["large", "medium", "small"]:
    for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
        _maps.append(f"varied_terrain/{terrain}_{size}")

for map_dir in _maps:
    short_name = map_dir.replace("varied_terrain/", "")
    NAVIGATION_MISSIONS.append(NavigationDenseMission(name=f"navigation_{short_name}", terrain_dir=map_dir))
