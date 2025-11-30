"""Navigation missions for CoGames.

These missions wrap the navigation environments defined in recipes.experiment.navigation
so they can be used within the standard Mission/Curriculum infrastructure.
"""

from typing import override

from cogames.cogs_vs_clips.mission import Mission
from mettagrid.config import vibes
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from metta.map.terrain_from_numpy import NavigationFromNumpy
from recipes.experiment import navigation


def _cleanup_nav_env(env: MettaGridConfig) -> MettaGridConfig:
    """Cleanup navigation environment to be compatible with CVC training.
    
    Ensures the environment configuration satisfies CVC constraints and matches
    the standard action space (using TRAINING_VIBES) to allow joint training.
    """
    # 1. Standardize Vibe Configuration
    # Use TRAINING_VIBES (subset) to match the curriculum standard.
    env.game.vibe_names = [vibe.name for vibe in vibes.TRAINING_VIBES]
    
    if env.game.actions:
        # 2. Ensure Vibe Action matches
        if env.game.actions.change_vibe:
            env.game.actions.change_vibe.enabled = True
            env.game.actions.change_vibe.number_of_vibes = len(vibes.TRAINING_VIBES)
            
            # Filter initial vibe to be within range
            if env.game.agent.initial_vibe >= len(vibes.TRAINING_VIBES):
                env.game.agent.initial_vibe = 0

        # 3. Force Enable Other Actions (Attack, Resource Mod)
        # CVC missions typically have these enabled. We must match the action space.
        # Even if they are useless in navigation, they must be present in the schema.
        if env.game.actions.attack:
            env.game.actions.attack.enabled = True
            # Ensure target locations match CVC defaults if needed (usually 1-9)
            if not env.game.actions.attack.target_locations:
                 env.game.actions.attack.target_locations = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        if env.game.actions.resource_mod:
            env.game.actions.resource_mod.enabled = True

    return env


class NavigationMission(Mission):
    """A mission that wraps a navigation environment builder."""

    nav_map_name: str
    max_steps: int = 1000
    # Set num_instances to 1 to align with num_cogs being the total agent count
    num_instances: int = 1

    def __init__(self, name: str, nav_map_name: str, **kwargs):
        # We pass a dummy site because Mission requires one, but we'll override make_env
        # so it won't be used.
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name=name,
            description=f"Navigation task on map {nav_map_name}",
            site=HELLO_WORLD,
            nav_map_name=nav_map_name,  # Pass field to Pydantic init
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        # Delegate to the navigation recipe's builder
        # Note: navigation.make_nav_ascii_env expects num_agents per instance
        # Fallback to 1 agent if not set (though it should be)
        num_agents = self.num_cogs if self.num_cogs is not None else 1
        
        env = navigation.make_nav_ascii_env(
            name=self.nav_map_name,
            max_steps=self.max_steps,
            num_agents=num_agents,
            num_instances=self.num_instances,
        )
        return _cleanup_nav_env(env)


class SparseNavigationMission(Mission):
    """A mission that wraps the sparse empty space navigation environment."""

    def __init__(self, **kwargs):
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name="emptyspace_sparse",
            description="Navigation in sparse empty space",
            site=HELLO_WORLD,
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        # make_emptyspace_sparse_env defaults to 4 agents.
        # We should ideally respect num_cogs, but the helper might not support it.
        # For now, we accept the default or modify if needed.
        env = navigation.make_emptyspace_sparse_env()
        
        # If num_cogs is set, force update the agent count
        if self.num_cogs:
            env.game.num_agents = self.num_cogs
            # Also update map builder if it has agent count
            if isinstance(env.game.map_builder, RandomMapBuilder.Config): # It might be MapGen wrapping Random
                 pass # Handling nested config is complex, trusting default for sparse eval
                 
        return _cleanup_nav_env(env)


class NavigationDenseMission(Mission):
    """A mission that uses the dense varied terrain maps from navigation training.
    
    This uses a specific 'varied_terrain' directory configuration.
    """
    
    terrain_dir: str

    def __init__(self, name: str, terrain_dir: str, **kwargs):
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name=name,
            description=f"Dense varied terrain navigation: {terrain_dir}",
            site=HELLO_WORLD,
            terrain_dir=terrain_dir,  # Pass field to Pydantic init
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        num_agents = self.num_cogs if self.num_cogs is not None else 1
        
        # Base navigation setup
        # Use 1 instance to match num_cogs as total agents
        nav_env = navigation.mettagrid(num_agents=num_agents, num_instances=1)
        
        # Override map builder instance with the specific directory
        # We check structure via assertion instead of cast
        map_builder = nav_env.game.map_builder
        assert isinstance(map_builder, MapGen.Config), "Expected MapGen.Config for navigation map builder"
        
        default_instance = map_builder.instance
        if isinstance(default_instance, NavigationFromNumpy.Config):
            objects = default_instance.objects
        else:
            objects = {"altar": 10}

        map_builder.instance = NavigationFromNumpy.Config(
            agents=num_agents,
            objects=objects,
            dir=self.terrain_dir,
        )
        
        return _cleanup_nav_env(nav_env)


class NavigationSparseRandomMission(Mission):
    """A mission that uses random sparse maps from navigation training.
    """

    def __init__(self, **kwargs):
        from cogames.cogs_vs_clips.sites import HELLO_WORLD

        super().__init__(
            name="navigation_sparse_random",
            description="Random sparse navigation maps",
            site=HELLO_WORLD,
            **kwargs,
        )

    @override
    def make_env(self) -> MettaGridConfig:
        num_agents = self.num_cogs if self.num_cogs is not None else 1
        
        # Use 1 instance
        nav_env = navigation.mettagrid(num_agents=num_agents, num_instances=1)
        
        # Override with RandomMapBuilder
        nav_env.game.map_builder = RandomMapBuilder.Config(
            agents=num_agents, # Total agents
            objects={"altar": 10},
            width=100,
            height=100,
        )
        return _cleanup_nav_env(nav_env)


# 1. Navigation Sequence Missions (from eval config)
NAVIGATION_MISSIONS: list[Mission] = [
    NavigationMission(
        name=f"navigation_{eval_cfg['name']}",
        nav_map_name=eval_cfg['name'],
        max_steps=eval_cfg['max_steps'],
        num_cogs=eval_cfg['num_agents'],
        num_instances=1 # Override to 1 for compatibility
    )
    for eval_cfg in navigation.NAVIGATION_EVALS
]

# 2. Sparse Mission (static eval)
NAVIGATION_MISSIONS.append(SparseNavigationMission())

# 3. Training Missions (Varied Terrain configurations)
# From recipes.experiment.navigation.make_curriculum
_maps = []
for size in ["large", "medium", "small"]:
    for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
        _maps.append(f"varied_terrain/{terrain}_{size}")

# Add a mission for each map configuration
for map_dir in _maps:
    # map_dir e.g. "varied_terrain/dense_large"
    # Name it "navigation_dense_large"
    short_name = map_dir.replace("varied_terrain/", "")
    NAVIGATION_MISSIONS.append(
        NavigationDenseMission(
            name=f"navigation_{short_name}",
            terrain_dir=map_dir
        )
    )

# Add the sparse random one too
NAVIGATION_MISSIONS.append(NavigationSparseRandomMission())
