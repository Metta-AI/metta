"""Navigation evaluation suite using procedural corridor generation.

This replaces static ASCII maps with dynamically generated corridor patterns
using the unified AngledCorridorBuilder API.
"""

from typing import List, Optional

import numpy as np
from metta.map.scenes.angled_corridor_builder import (
    AngledCorridorBuilder,
    AngledCorridorBuilderParams,
    horizontal,
    vertical,
)
from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.map_builder.ops_builder import OpsMapBuilder
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.ops import (
    Operation,
    StampOp,
    grid_ops,
    line,
    radial_ops,
    react,
)
from metta.mettagrid.mapgen.scenes.mean_distance import MeanDistance
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def _stamp_cell(y: int, x: int, material: str) -> StampOp:
    import numpy as _np

    return StampOp(center=(y, x), pattern=_np.array([[material]], dtype=str))


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


def make_corridors_env(vertical_orientation: bool = False) -> MettaGridConfig:
    """Generate corridors pattern using ops.

    Args:
        vertical_orientation: If True, vertical main and horizontal cross-cuts
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 450

    from random import choice

    if vertical_orientation:
        width, height = 26, 74
        ops: List[Operation] = []
        # Main vertical
        ops.append(line((1, 13), (73, 13), thickness=7))
        # Cross-cuts
        ops.append(line((10, 1), (10, 10), thickness=3))
        ops.append(line((20, 1), (20, 25), thickness=4))
        ops.append(line((30, 17), (30, 25), thickness=2))
        ops.append(line((40, 1), (40, 25), thickness=3))
        ops.append(line((50, 5), (50, 20), thickness=5))
        ops.append(line((60, 1), (60, 25), thickness=2))
        ops.append(line((68, 8), (68, 18), thickness=3))
        _ = choice([2, 72])  # maintain source of randomness; agent placed by env
        # Place altars near far cross-cuts
        ops.append(_stamp_cell(10, 9, "altar"))
        ops.append(_stamp_cell(50, 19, "altar"))
        ops.append(_stamp_cell(68, 17, "altar"))
        env.game.map_builder = MapGen.Config(
            instances=4,
            border_width=6,
            instance_border_width=3,
            instance_map=OpsMapBuilder.Config(
                ops=ops,
                width=width,
                height=height,
                initial_fill="wall",
                border_width=2,
            ),
        )
    else:
        width, height = 74, 26
        ops = []
        # Main horizontal
        ops.append(line((13, 1), (13, 73), thickness=7))
        # Cross-cuts
        ops.append(line((1, 10), (10, 10), thickness=3))
        ops.append(line((1, 20), (25, 20), thickness=4))
        ops.append(line((17, 30), (25, 30), thickness=2))
        ops.append(line((1, 40), (25, 40), thickness=3))
        ops.append(line((5, 50), (20, 50), thickness=5))
        ops.append(line((1, 60), (25, 60), thickness=2))
        ops.append(line((8, 68), (18, 68), thickness=3))
        _ = choice([2, 72])
        # Altars near far cross-cuts
        ops.append(_stamp_cell(9, 10, "altar"))
        ops.append(_stamp_cell(19, 50, "altar"))
        ops.append(_stamp_cell(17, 68, "altar"))
        env.game.map_builder = MapGen.Config(
            instances=4,
            border_width=6,
            instance_border_width=3,
            instance_map=OpsMapBuilder.Config(
                ops=ops,
                width=width,
                height=height,
                initial_fill="wall",
                border_width=2,
            ),
        )

    return make_nav_eval_env(env)


def make_radial_mini_env() -> MettaGridConfig:
    """Generate radial_mini using ops-based radial spokes."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 150

    center = (11, 11)
    ops: List[Operation] = []
    ops.extend(radial_ops(center=center, num_spokes=8, length=8, thickness=1))
    # Place altars at three distinct spoke ends deterministically
    ops.append(_stamp_cell(center[0] - 8, center[1], "altar"))
    ops.append(_stamp_cell(center[0], center[1] + 8, "altar"))
    ops.append(_stamp_cell(center[0] + 8, center[1], "altar"))

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=ops,
            width=22,
            height=22,
            initial_fill="wall",
            border_width=1,
        ),
    )
    return make_nav_eval_env(env)


def make_radial_small_env() -> MettaGridConfig:
    """Generate radial_small using ops-based radial spokes."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 120

    center = (17, 17)
    ops: List[Operation] = []
    ops.extend(radial_ops(center=center, num_spokes=6, length=12, thickness=2))
    ops.append(_stamp_cell(center[0] - 12, center[1], "altar"))
    ops.append(_stamp_cell(center[0] + 12, center[1], "altar"))
    ops.append(_stamp_cell(center[0], center[1] + 12, "altar"))

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=ops,
            width=34,
            height=34,
            initial_fill="wall",
            border_width=2,
        ),
    )
    return make_nav_eval_env(env)


def make_radial_large_env() -> MettaGridConfig:
    """Generate radial_large using ops-based star spokes."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 1000

    center = (40, 40)
    ops: List[Operation] = []
    # Star = pairwise opposite radial lines; approximate with 8 spokes
    ops.extend(radial_ops(center=center, num_spokes=8, length=35, thickness=3))
    # One altar at a far end
    ops.append(_stamp_cell(center[0] - 35, center[1], "altar"))

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=ops,
            width=80,
            height=80,
            initial_fill="wall",
            border_width=3,
        ),
    )
    return make_nav_eval_env(env)


def make_grid_maze_env() -> MettaGridConfig:
    """Generate a grid/maze pattern using ops."""
    env = make_navigation(num_agents=4)
    env.game.max_steps = 500

    ops: List[Operation] = []
    # Base grid
    ops.extend(grid_ops(origin=(10, 10), rows=5, cols=5, spacing=10, thickness=2))
    # Altar stamps at four intersections
    for y in [10, 30, 50, 30]:
        for x in [10, 30, 50, 30]:
            if (y, x) in [(10, 10), (10, 50), (50, 10), (50, 50)]:
                ops.append(_stamp_cell(y, x, "altar"))

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=ops,
            width=60,
            height=60,
            initial_fill="wall",
            border_width=2,
        ),
    )
    return make_nav_eval_env(env)


def make_hall_of_mirrors_env() -> MettaGridConfig:
    """Procedural version of hall_of_mirrors using mirrored corridors.

    Creates a central horizontal hall with symmetric vertical pillars above/below
    and short upper/lower bands to evoke reflective symmetry.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 400

    width = 29
    height = 27
    center_y = height // 2
    center_x = width // 2

    corridors = []

    # Central hall
    corridors.append(horizontal(y=center_y, thickness=3, x_start=1, x_end=width - 2))

    # Symmetric vertical pillars (split above and below the center hall)
    for dx in (5, 9):
        left_x = center_x - dx
        right_x = center_x + dx
        corridors.append(vertical(x=left_x, thickness=2, y_start=1, y_end=center_y - 1))
        corridors.append(
            vertical(x=right_x, thickness=2, y_start=center_y + 2, y_end=height - 2)
        )

    # Upper and lower bands
    corridors.append(horizontal(y=3, thickness=1, x_start=3, x_end=width - 3))
    corridors.append(horizontal(y=height - 4, thickness=1, x_start=3, x_end=width - 3))

    params = AngledCorridorBuilderParams(
        corridors=corridors,
        objects={"altar": 6},
        place_at_ends=True,
        prefer_far_from_center=True,
        agent_position=(center_y, 2),
        shuffle_placements=False,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=width,
            height=height,
            border_width=1,
            root=AngledCorridorBuilder.factory(params=params),
        ),
    )

    return make_nav_eval_env(env)


def make_rooms_env() -> MettaGridConfig:
    """Procedural remake of systematic_exploration_memory/rooms.map using stamping walls.

    We start from an empty canvas and carve rectangular rooms, automatically
    surrounding them with 1-cell wall rings. Doorways are carved by additional
    thin corridors that cross the ring.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 400

    # Approximate the original map dimensions (inner area)
    width = 38
    height = 45

    corridors = []

    # Room A (top-left): bounded ring with an open doorway (post-carve)
    corridors.append(
        horizontal(y=9, thickness=9, x_start=4, x_end=15, surround_with_walls=True)
    )
    # Room B (mid-right)
    corridors.append(
        horizontal(y=22, thickness=11, x_start=24, x_end=36, surround_with_walls=True)
    )
    # Room C (bottom-left)
    corridors.append(
        horizontal(y=38, thickness=11, x_start=3, x_end=20, surround_with_walls=True)
    )

    # Doorways that should not be surrounded
    post_doors = [
        vertical(x=6, thickness=1, y_start=1, y_end=12, surround_with_walls=False),
        horizontal(y=22, thickness=1, x_start=18, x_end=24, surround_with_walls=False),
        vertical(x=10, thickness=1, y_start=30, y_end=38, surround_with_walls=False),
    ]

    params = AngledCorridorBuilderParams(
        corridors=corridors,
        post_carve_corridors=post_doors,
        # Start empty and surround carved areas with wall rings
        initial_fill="empty",
        surround_with_walls=False,
        wall_sides=["N", "S", "E", "W"],
        # Place a few altars at fixed inner coords to avoid blocking
        fixed_objects={"altar": [(9, 8), (22, 30), (38, 10)]},
        # Prefer center placement after fixed objects only for any remaining objects
        objects={},
        place_at_center=False,
        place_at_ends=False,
        agent_position=(9, 8),  # inside Room A
        shuffle_placements=False,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=width,
            height=height,
            border_width=1,
            root=AngledCorridorBuilder.factory(params=params),
        ),
    )

    return make_nav_eval_env(env)


def make_tease_small_env() -> MettaGridConfig:
    """Procedural remake of systematic_exploration_memory/tease_small.map.

    Large left room bounded by a wall ring with one bottom-right opening,
    plus scattered altars and agent in the top-right.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 300

    width = 72
    height = 17

    corridors = []

    # Main left room (rectangle via thick horizontal corridor)
    corridors.append(
        horizontal(y=9, thickness=7, x_start=8, x_end=60, surround_with_walls=True)
    )

    # Bottom-right opening: carve across the ring
    corridors.append(
        horizontal(y=14, thickness=1, x_start=56, x_end=69, surround_with_walls=False)
    )

    # Interior decoration corridors for mild asymmetry (optional)
    corridors.append(
        vertical(x=12, thickness=1, y_start=5, y_end=12, surround_with_walls=False)
    )

    params = AngledCorridorBuilderParams(
        corridors=corridors,
        initial_fill="empty",
        surround_with_walls=False,
        wall_sides=["N", "S", "E", "W"],
        fixed_objects={"altar": [(14, 63), (5, 6), (6, 10)]},
        objects={},
        place_at_center=False,
        place_at_ends=False,
        agent_position=(2, width - 3),  # near top-right like the ASCII
        shuffle_placements=False,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=width,
            height=height,
            border_width=1,
            root=AngledCorridorBuilder.factory(params=params),
        ),
    )

    return make_nav_eval_env(env)


def make_tease_env() -> MettaGridConfig:
    """Procedural remake of systematic_exploration_memory/tease.map.

    Same motif as tease_small, but taller, with additional lower features.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 500

    width = 72
    height = 44

    corridors = []

    # Main left room (taller variant)
    corridors.append(
        horizontal(y=22, thickness=15, x_start=8, x_end=60, surround_with_walls=True)
    )

    # Bottom-right opening of the room ring
    corridors.append(
        horizontal(y=32, thickness=1, x_start=54, x_end=69, surround_with_walls=False)
    )

    # A mid-height thin connector (echoing the long corridor edge)
    corridors.append(
        horizontal(y=28, thickness=1, x_start=16, x_end=24, surround_with_walls=False)
    )

    params = AngledCorridorBuilderParams(
        corridors=corridors,
        initial_fill="empty",
        surround_with_walls=False,
        wall_sides=["N", "S", "E", "W"],
        fixed_objects={"altar": [(40, 65), (8, 6), (10, 12)]},
        objects={},
        place_at_center=False,
        place_at_ends=False,
        agent_position=(2, width - 3),
        shuffle_placements=False,
    )

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config(
            width=width,
            height=height,
            border_width=1,
            root=AngledCorridorBuilder.factory(params=params),
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


def make_ops_demo_env() -> MettaGridConfig:
    """Demonstrate pure operations-based map generation.

    This creates a map using only operations, showing how the chemistry
    approach can work alongside the existing corridor builder.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 400

    # Generate operations for a combined pattern
    ops = []

    # Add a grid base
    ops.extend(grid_ops(origin=(10, 10), rows=5, cols=5, spacing=8, thickness=2))

    # Add radial patterns at grid intersections
    for i in range(0, 6):
        for j in range(0, 6):
            center = (10 + i * 8, 10 + j * 8)
            if (i + j) % 2 == 0:  # Checkerboard pattern
                ops.extend(
                    radial_ops(center=center, num_spokes=4, length=3, thickness=1)
                )

    # Use OpsMapBuilder directly
    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=ops, width=60, height=60, initial_fill="empty", border_width=2
        ),
    )

    return make_nav_eval_env(env)


def make_chemistry_demo_env() -> MettaGridConfig:
    """Demonstrate chemistry reactions between operation sets.

    This shows how complex patterns emerge from reacting simple patterns.
    """
    env = make_navigation(num_agents=4)
    env.game.max_steps = 500

    # Create two sets of operations
    # Set 1: Diagonal lines
    diagonal_ops: List[Operation] = [
        line((10, 10), (40, 40), thickness=2),
        line((10, 40), (40, 10), thickness=2),
    ]

    # Set 2: Grid
    grid_pattern: List[Operation] = list(
        grid_ops(origin=(15, 15), rows=3, cols=3, spacing=10, thickness=1)
    )

    # React them with intersection rule - adds stamps at crossing points
    combined = react(diagonal_ops, grid_pattern, "intersection")

    # Add some radial patterns
    radial_pattern: List[Operation] = list(
        radial_ops(center=(30, 30), num_spokes=8, length=15, thickness=1)
    )

    # React with the combined pattern
    final_ops = react(combined, radial_pattern, "intersection")

    env.game.map_builder = MapGen.Config(
        instances=4,
        border_width=6,
        instance_border_width=3,
        instance_map=OpsMapBuilder.Config(
            ops=final_ops, width=60, height=60, initial_fill="empty", border_width=2
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
        SimulationConfig(
            name="corridors_vertical", env=make_corridors_env(vertical_orientation=True)
        ),
        SimulationConfig(name="radial_mini", env=make_radial_mini_env()),
        SimulationConfig(name="radial_small", env=make_radial_small_env()),
        SimulationConfig(name="radial_large", env=make_radial_large_env()),
        SimulationConfig(name="grid_maze", env=make_grid_maze_env()),
        SimulationConfig(name="rooms", env=make_rooms_env()),
        SimulationConfig(name="tease_small", env=make_tease_small_env()),
        SimulationConfig(name="tease", env=make_tease_env()),
        SimulationConfig(name="hall_of_mirrors", env=make_hall_of_mirrors_env()),
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
        # Pure operations demo
        SimulationConfig(name="ops_demo", env=make_ops_demo_env()),
        SimulationConfig(name="chemistry_demo", env=make_chemistry_demo_env()),
    ]
