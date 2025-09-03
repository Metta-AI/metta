"""Navigation evaluation suite with composable map transformations.

This module generates, per ASCII map, a set of transformed variants
without requiring manual enumeration in the return list.

Example naming per map:
  knotty -> {knotty, knotty_90, knotty_180, knotty_270, knotty_hflip, knotty_vflip, knotty_sx2, knotty_sy2, knotty_sxy2}

You can choose which transformation families to include by passing
`transform_set` and whether to include combinations (`transform_combo`).
By default we include a single transformation per map (no combos).
"""

import atexit
import os
import tempfile

from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.scenes.mean_distance import MeanDistance
from metta.mettagrid.mapgen.utils.ascii_transform import apply_transformations
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

# Currently we are transforming every map with all transformations (three rotations and two flips)
EVALS = {
    "corridors": 450,
    "cylinder_easy": 250,
    "cylinder": 250,
    "honeypot": 300,
    "knotty": 500,
    "memory_palace": 200,
    "obstacles0": 100,
    "obstacles1": 300,
    "obstacles2": 350,
    "obstacles3": 300,
    "radial_large": 1000,
    "radial_mini": 150,
    "radial_small": 120,
    "radial_maze": 200,
    "swirls": 350,
    "thecube": 350,
    "walkaround": 250,
    "wanderout": 500,
    "emptyspace_outofsight": 150,
    "walls_outofsight": 250,
    "walls_withinsight": 120,
    "labyrinth": 250,
    "boxout": 150,
    "choose_wisely": 200,
    "corners": 300,
    "hall_of_mirrors": 150,
    "journey_home": 110,
    "little_landmark_hard": 100,
    "lobster_legs_cues": 210,
    "lobster_legs": 210,
    "memory_swirls_hard": 300,
    "memory_swirls": 300,
    "passing_things": 320,
    "rooms": 350,
    "spacey_memory": 250,
    "spiral_chamber": 300,
    "tease_small": 300,
    "tease": 300,
    "venture_out": 300,
    "you_shall_not_pass": 120,
    "easy_sequence": 42,
    "medium_sequence": 58,
    "hard_sequence": 70,
}


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def replace_objects_with_altars(name: str) -> str:
    print(f"Replacing objects with altars for {name}")
    ascii_map = f"mettagrid/configs/maps/navigation_sequence/{name}.map"

    with open(ascii_map, "r") as f:
        map_content = f.read()

    map_content = map_content.replace("n", "_").replace("m", "_")

    with tempfile.NamedTemporaryFile(suffix=".map", mode="w", delete=False) as tmp:
        tmp.write(map_content)
        ascii_map_nav = tmp.name
    atexit.register(lambda p=ascii_map_nav: os.path.exists(p) and os.remove(p))

    return ascii_map_nav


def make_nav_ascii_env(
    name: str, max_steps: int, border_width: int = 1, num_agents=4
) -> MettaGridConfig:
    # we re-use nav sequence maps, but replace all objects with altars
    ascii_map = replace_objects_with_altars(name)

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
    """Create the navigation evaluation suite with per-map transformed variants.

    For each ASCII map, include: original + all single transforms from
    selected families (rotation, mirror, stretch). Combinations are disabled
    by default to limit explosion of variants.
    """

    sims: list[SimulationConfig] = []

    temp_files = []

    for eval_name, max_steps in EVALS.items():
        sims.append(
            SimulationConfig(
                name=eval_name, env=make_nav_ascii_env(eval_name, max_steps)
            )
        )
        for transform_name in apply_transformations(
            "mettagrid/configs/maps/navigation_sequence",
            eval_name,
            transform_set="all",
        ):
            # print(f"Appending the transformed, {transform_name}")
            sims.append(
                SimulationConfig(
                    name=f"{eval_name}_{transform_name}",
                    env=make_nav_ascii_env(transform_name, max_steps),
                )
            )
            temp_files.append(
                f"mettagrid/configs/maps/navigation_sequence/{transform_name}.map"
            )

    for temp_file in temp_files:
        os.remove(temp_file)

    # Non-ASCII procedural env left unchanged
    sims.append(
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env())
    )

    return sims
