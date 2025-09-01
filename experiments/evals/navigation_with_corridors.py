"""Navigation evaluation suite using procedural corridor generation.

This replaces static ASCII maps with dynamically generated corridor patterns
using the unified AngledCorridorBuilder API.
"""

from typing import Optional

import numpy as np
from metta.map.scenes.angled_corridor_builder import (
    AngledCorridorBuilder,
    AngledCorridorBuilderParams,
    horizontal,
    radial_corridors,
    star_pattern,
    vertical,
)
from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.scenes.mean_distance import MeanDistance
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_nav_ascii_env(
    name: str, max_steps: int, border_width: int = 1, num_agents=4
) -> MettaGridConfig:
    """Load navigation env from ASCII map file."""
    ascii_map = f"mettagrid/configs/maps/navigation/{name}.map"
    env = make_navigation(num_agents=num_agents)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config.with_ascii_uri(ascii_map, border_width=border_width),
    )
    return make_nav_eval_env(env)


def make_corridors_env() -> MettaGridConfig:
    """Generate corridors.map pattern procedurally."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 450

    # Recreate corridors.map pattern
    corridors = [
        horizontal(y=13, thickness=7, x_start=1, x_end=73),
        vertical(x=10, thickness=3, y_start=1, y_end=10),
        vertical(x=20, thickness=4),
        vertical(x=30, thickness=2, y_start=17, y_end=26),
        vertical(x=40, thickness=3),
        vertical(x=50, thickness=5, y_start=5, y_end=20),
        vertical(x=60, thickness=2),
        vertical(x=68, thickness=3, y_start=8, y_end=18),
    ]

    # Randomize agent at either end of main horizontal corridor
    from random import choice

    agent_x = choice([2, 72])

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=74,
            height=26,
            border_width=2,
            root=AngledCorridorBuilder.factory(
                params=AngledCorridorBuilderParams(
                    corridors=corridors,
                    objects={"altar": 3},
                    place_at_ends=True,
                    agent_position=(13, agent_x),
                    ensure_altar_near_agent=False,
                    prefer_far_from_center=True,
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_radial_mini_env() -> MettaGridConfig:
    """Generate radial_mini.map pattern procedurally."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 150

    corridors = radial_corridors(
        center=(11, 11),
        num_spokes=8,
        length=8,
        thickness=1,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=22,
            height=22,
            border_width=1,
            root=AngledCorridorBuilder.factory(
                params=AngledCorridorBuilderParams(
                    corridors=corridors,
                    objects={"altar": 3},
                    place_at_ends=True,
                    agent_position=(11, 11),
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_radial_small_env() -> MettaGridConfig:
    """Generate radial_small.map pattern procedurally."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 120

    corridors = radial_corridors(
        center=(17, 17),
        num_spokes=6,
        length=12,
        thickness=2,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=34,
            height=34,
            border_width=2,
            root=AngledCorridorBuilder.factory(
                params=AngledCorridorBuilderParams(
                    corridors=corridors,
                    objects={"altar": 3},
                    place_at_ends=True,
                    agent_position=(17, 17),
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_radial_large_env() -> MettaGridConfig:
    """Generate radial_large.map pattern procedurally."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 1000

    # Create a star pattern for larger map
    corridors = star_pattern(
        center=(40, 40),
        num_spokes=8,
        length=35,
        thickness=3,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=80,
            height=80,
            border_width=3,
            root=AngledCorridorBuilder.factory(
                params=AngledCorridorBuilderParams(
                    corridors=corridors,
                    objects={"altar": 1},
                    place_at_ends=True,
                    agent_position=(40, 40),
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_grid_maze_env() -> MettaGridConfig:
    """Generate a grid/maze pattern using corridors."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 500

    corridors = []

    # Create a grid pattern
    for y in [10, 20, 30, 40, 50]:
        corridors.append(horizontal(y=y, thickness=2))

    for x in [10, 20, 30, 40, 50]:
        corridors.append(vertical(x=x, thickness=2))

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=60,
            height=60,
            border_width=2,
            root=AngledCorridorBuilder.factory(
                params=AngledCorridorBuilderParams(
                    corridors=corridors,
                    objects={"altar": 4},
                    place_at_intersections=True,
                    agent_position=(30, 30),
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_hard_sequence_env() -> MettaGridConfig:
    """Generate a procedural version of hard_sequence.map.

    Pattern:
      - Three 3-wide vertical lanes separated by single wall columns
      - Two stages separated by a solid horizontal wall band
      - Altars placed at the lower ends; agent at the center lower end of stage 1
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 600

    width = 13
    height = 33

    # Column centers for three 3-wide lanes inside a 13-wide map
    # Layout: # [1..3] # [5..7] # [9..11] #  -> centers at 2, 6, 10
    lane_centers = [2, 6, 10]

    # Top horizontal hallway
    top_y = 5

    corridors = []
    corridors.append(horizontal(y=top_y, thickness=3, x_start=1, x_end=width - 2))
    # Three vertical lanes starting at the top band to intersect it
    for x in lane_centers:
        corridors.append(vertical(x=x, thickness=3, y_start=top_y, y_end=height - 2))

    # Place altars at lower ends; agent at the bottom of the middle hallway
    params = AngledCorridorBuilderParams(
        corridors=corridors,
        objects={"altar": 3},
        place_at_ends=True,
        prefer_lower_ends=True,
        agent_position=(height - 2, lane_centers[1]),
        shuffle_placements=False,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=width,
            height=height,
            border_width=0,
            root=AngledCorridorBuilder.factory(params=params),
        ),
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> MettaGridConfig:
    """Keep the original mean distance implementation."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 300
    env.game.map_builder = MapGen.Config(
        instances=4,
        instance_map=MapGen.Config(
            width=60,
            height=60,
            border_width=3,
            root=MeanDistance.factory(
                params=MeanDistance.Params(
                    mean_distance=30,
                    objects={"altar": 3},
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def visualize_env_map(
    env_func, title: Optional[str] = None, show_ascii: bool = True
) -> np.ndarray:
    """Helper function to visualize a map from an environment function.

    Args:
        env_func: Function that returns an EnvConfig (e.g., make_corridors_env)
        title: Optional title for the visualization
        show_ascii: Whether to print ASCII representation

    Returns:
        The generated map grid as numpy array
    """
    # Generate the environment config
    env_config = env_func()

    # Extract the instance map configuration
    instance_map_config = env_config.game.map_builder.instance_map

    # Build the map using MapGen
    map_gen = MapGen(instance_map_config)
    game_map = map_gen.build()
    grid = game_map.grid

    if show_ascii:
        # Print ASCII representation
        if title:
            print(f"\n{'=' * 60}")
            print(f"{title} ({grid.shape[0]}x{grid.shape[1]})")
            print("=" * 60)

        char_map = {
            "wall": "#",
            "empty": ".",
            "agent.agent": "@",
            "altar": "_",
            "heart": "♥",
        }

        for i, row in enumerate(grid):
            if i >= 40:  # Limit height for display
                print(f"... ({grid.shape[0] - 40} more rows)")
                break
            line = ""
            for j, cell in enumerate(row):
                if j >= 80:  # Limit width for display
                    line += "..."
                    break
                line += char_map.get(cell, "?")
            print(line)

    return grid


def make_navigation_eval_suite() -> list[SimulationConfig]:
    """Navigation evaluation suite with procedural corridor generation.

    This suite replaces static ASCII maps with dynamically generated
    corridor patterns where appropriate, while keeping some ASCII maps
    for specific unique patterns.
    """
    return [
        # Procedurally generated corridor maps
        SimulationConfig(name="corridors", env=make_corridors_env()),
        SimulationConfig(name="radial_mini", env=make_radial_mini_env()),
        SimulationConfig(name="radial_small", env=make_radial_small_env()),
        SimulationConfig(name="radial_large", env=make_radial_large_env()),
        SimulationConfig(name="grid_maze", env=make_grid_maze_env()),
        SimulationConfig(name="hard_sequence", env=make_hard_sequence_env()),
        # ASCII maps that are unique enough to keep as-is
        SimulationConfig(
            name="cylinder_easy", env=make_nav_ascii_env("cylinder_easy", 250)
        ),
        SimulationConfig(name="cylinder", env=make_nav_ascii_env("cylinder", 250)),
        SimulationConfig(name="honeypot", env=make_nav_ascii_env("honeypot", 300)),
        SimulationConfig(name="knotty", env=make_nav_ascii_env("knotty", 500)),
        SimulationConfig(
            name="memory_palace", env=make_nav_ascii_env("memory_palace", 200)
        ),
        SimulationConfig(name="obstacles0", env=make_nav_ascii_env("obstacles0", 100)),
        SimulationConfig(name="obstacles1", env=make_nav_ascii_env("obstacles1", 300)),
        SimulationConfig(name="obstacles2", env=make_nav_ascii_env("obstacles2", 350)),
        SimulationConfig(name="obstacles3", env=make_nav_ascii_env("obstacles3", 300)),
        SimulationConfig(
            name="radial_maze", env=make_nav_ascii_env("radial_maze", 200)
        ),
        SimulationConfig(name="swirls", env=make_nav_ascii_env("swirls", 350)),
        SimulationConfig(name="thecube", env=make_nav_ascii_env("thecube", 350)),
        SimulationConfig(name="walkaround", env=make_nav_ascii_env("walkaround", 250)),
        SimulationConfig(name="wanderout", env=make_nav_ascii_env("wanderout", 500)),
        SimulationConfig(
            name="emptyspace_outofsight",
            env=make_nav_ascii_env("emptyspace_outofsight", 150),
        ),
        SimulationConfig(
            name="walls_outofsight", env=make_nav_ascii_env("walls_outofsight", 250)
        ),
        SimulationConfig(
            name="walls_withinsight", env=make_nav_ascii_env("walls_withinsight", 120)
        ),
        SimulationConfig(name="labyrinth", env=make_nav_ascii_env("labyrinth", 250)),
        # Mean distance based
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env()),
    ]
