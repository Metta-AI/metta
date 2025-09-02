"""Navigation evaluation suite with map transformations.

This is a variant of navigation.py that applies transformations (rotation, mirroring, stretching)
to the ASCII maps. Each transformation can be enabled/disabled via the configuration flags below.
"""

from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.scenes.mean_distance import MeanDistance
from metta.mettagrid.mapgen.utils.ascii_transform import (
    mirror_ascii_map,
    rotate_ascii_map,
    stretch_ascii_map,
)
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

# ============================================================================
# TRANSFORMATION CONFIGURATION
# Set these flags to control which transformations are applied to the maps
# ============================================================================

# Rotation (clockwise)
ENABLE_ROTATE_90 = False  # Rotate maps 90 degrees clockwise
ENABLE_ROTATE_180 = False  # Rotate maps 180 degrees
ENABLE_ROTATE_270 = False  # Rotate maps 270 degrees

# Mirroring
ENABLE_MIRROR_HORIZONTAL = False  # Mirror left-right
ENABLE_MIRROR_VERTICAL = False  # Mirror top-bottom

# Stretching (2x scale)
ENABLE_STRETCH_HORIZONTAL = False  # Double width (scale_x=2)
ENABLE_STRETCH_VERTICAL = False  # Double height (scale_y=2)
ENABLE_STRETCH_BOTH = False  # Double both dimensions (scale_x=2, scale_y=2)

# ============================================================================


def _apply_transformations(ascii_map_path: str) -> str:
    """Load and optionally transform an ASCII map based on configuration flags.

    Args:
        ascii_map_path: Path to the ASCII map file

    Returns:
        Transformed ASCII map content as a string
    """
    # Read the original map
    with open(ascii_map_path, "r", encoding="utf-8") as f:
        map_content = f.read().strip()

    # Apply rotation if enabled
    if ENABLE_ROTATE_90:
        map_content = rotate_ascii_map(map_content, 90)
    elif ENABLE_ROTATE_180:
        map_content = rotate_ascii_map(map_content, 180)
    elif ENABLE_ROTATE_270:
        map_content = rotate_ascii_map(map_content, 270)

    # Apply mirroring if enabled
    if ENABLE_MIRROR_HORIZONTAL:
        map_content = mirror_ascii_map(map_content, "horizontal")
    if ENABLE_MIRROR_VERTICAL:
        map_content = mirror_ascii_map(map_content, "vertical")

    # Apply stretching if enabled (only one stretch mode at a time)
    if ENABLE_STRETCH_BOTH:
        map_content = stretch_ascii_map(map_content, scale_x=2, scale_y=2)
    elif ENABLE_STRETCH_HORIZONTAL:
        map_content = stretch_ascii_map(map_content, scale_x=2, scale_y=1)
    elif ENABLE_STRETCH_VERTICAL:
        map_content = stretch_ascii_map(map_content, scale_x=1, scale_y=2)

    return map_content


def _save_transformed_map(transformed_content: str, original_path: str) -> str:
    """Save transformed map to a temporary file and return its path.

    Args:
        transformed_content: The transformed ASCII map content
        original_path: Original map file path (for naming)

    Returns:
        Path to the temporary file containing the transformed map
    """
    import os
    import tempfile

    # Create a temporary file with a descriptive name
    base_name = os.path.basename(original_path)
    name_parts = base_name.rsplit(".", 1)
    transformed_name = (
        f"{name_parts[0]}_transformed.{name_parts[1]}"
        if len(name_parts) > 1
        else f"{base_name}_transformed"
    )

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".map",
        prefix=transformed_name.replace(".map", "_"),
        delete=False,
    ) as tf:
        tf.write(transformed_content)
        return tf.name


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_nav_ascii_env(
    name: str, max_steps: int, border_width: int = 1, num_agents=4
) -> MettaGridConfig:
    original_ascii_map = f"mettagrid/configs/maps/navigation/{name}.map"

    # Apply transformations if any are enabled
    any_transformations = any(
        [
            ENABLE_ROTATE_90,
            ENABLE_ROTATE_180,
            ENABLE_ROTATE_270,
            ENABLE_MIRROR_HORIZONTAL,
            ENABLE_MIRROR_VERTICAL,
            ENABLE_STRETCH_HORIZONTAL,
            ENABLE_STRETCH_VERTICAL,
            ENABLE_STRETCH_BOTH,
        ]
    )

    if any_transformations:
        transformed_content = _apply_transformations(original_ascii_map)
        ascii_map = _save_transformed_map(transformed_content, original_ascii_map)
    else:
        ascii_map = original_ascii_map

    env = make_navigation(num_agents=num_agents)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config.with_ascii_uri(ascii_map, border_width=border_width),
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> MettaGridConfig:
    """Note: This environment uses procedural generation, not ASCII maps, so transformations don't apply."""
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


def make_navigation_eval_suite() -> list[SimulationConfig]:
    """Create the navigation evaluation suite with optional transformations.

    Note: Transformations are applied to all ASCII map-based environments based on
    the configuration flags at the top of this file.
    """
    return [
        SimulationConfig(name="corridors", env=make_nav_ascii_env("corridors", 450)),
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
            name="radial_large", env=make_nav_ascii_env("radial_large", 1000)
        ),
        SimulationConfig(
            name="radial_mini", env=make_nav_ascii_env("radial_mini", 150)
        ),
        SimulationConfig(
            name="radial_small", env=make_nav_ascii_env("radial_small", 120)
        ),
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
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env()),
    ]
