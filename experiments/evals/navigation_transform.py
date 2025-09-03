"""Navigation evaluation suite with composable map transformations.

This module generates, per ASCII map, a set of transformed variants
without requiring manual enumeration in the return list.

Example naming per map:
  knotty -> {knotty, knotty_90, knotty_180, knotty_270, knotty_hflip, knotty_vflip, knotty_sx2, knotty_sy2, knotty_sxy2}

You can choose which transformation families to include by passing
`transform_set` and whether to include combinations (`transform_combo`).
By default we include a single transformation per map (no combos).
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

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


@dataclass(frozen=True)
class Transform:
    """Describes a single map transformation and its naming suffix."""

    name: str
    suffix: str
    apply: Callable[[str], str]  # takes map content, returns transformed content


# Families of single transformations (clockwise rotations, flips, and 2x stretch)
# Comment out transformations you don't want to use
ROTATIONS: list[Transform] = [
    Transform(name="rotate", suffix="90", apply=lambda s: rotate_ascii_map(s, 90)),
    Transform(name="rotate", suffix="180", apply=lambda s: rotate_ascii_map(s, 180)),
    Transform(name="rotate", suffix="270", apply=lambda s: rotate_ascii_map(s, 270)),
]

MIRRORS: list[Transform] = [
    Transform(
        name="mirror", suffix="hflip", apply=lambda s: mirror_ascii_map(s, "horizontal")
    ),
    Transform(
        name="mirror", suffix="vflip", apply=lambda s: mirror_ascii_map(s, "vertical")
    ),
]

STRETCHES: list[Transform] = [
    Transform(
        name="stretch",
        suffix="sx2",
        apply=lambda s: stretch_ascii_map(s, scale_x=2, scale_y=1),
    ),
    Transform(
        name="stretch",
        suffix="sy2",
        apply=lambda s: stretch_ascii_map(s, scale_x=1, scale_y=2),
    ),
    Transform(
        name="stretch",
        suffix="sxy2",
        apply=lambda s: stretch_ascii_map(s, scale_x=2, scale_y=2),
    ),
]


def _load_map_content(ascii_map_path: str) -> str:
    with open(ascii_map_path, "r", encoding="utf-8") as f:
        return f.read().strip()


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


def _build_env_from_ascii_path(
    ascii_path: str, max_steps: int, border_width: int = 1, num_agents: int = 4
) -> MettaGridConfig:
    env = make_navigation(num_agents=num_agents)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config.with_ascii_uri(
            ascii_path, border_width=border_width
        ),
    )
    return make_nav_eval_env(env)


def apply_transformations(
    name: str,
    max_steps: int,
    *,
    transform_set: Iterable[str] | str = "all",
    transform_combo: bool = False,
    include_original: bool = True,
    border_width: int = 1,
    num_agents: int = 4,
) -> list[SimulationConfig]:
    """Generate a list of SimulationConfigs for one map with selected transforms.

    - transform_set: "all" | Iterable of families ("rotation", "mirror", "stretch")
    - transform_combo: if True, also include pairwise combinations of families (off by default)
    - include_original: include the unmodified base map
    """
    original_ascii_map = f"mettagrid/configs/maps/navigation/{name}.map"
    original_content = _load_map_content(original_ascii_map)

    families: dict[str, list[Transform]] = {
        "rotation": ROTATIONS,
        "mirror": MIRRORS,
        "stretch": STRETCHES,
    }

    if transform_set == "all":
        selected_families = list(families.keys())
    else:
        selected_families = [fam for fam in transform_set if fam in families]

    sims: list[SimulationConfig] = []

    # Baseline (no transform)
    if include_original:
        sims.append(
            SimulationConfig(
                name=name,
                env=_build_env_from_ascii_path(
                    original_ascii_map, max_steps, border_width, num_agents
                ),
            )
        )

    # Single-family transforms
    for fam in selected_families:
        for t in families[fam]:
            transformed = t.apply(original_content)
            tmp_path = _save_transformed_map(transformed, original_ascii_map)
            sims.append(
                SimulationConfig(
                    name=f"{name}_{t.suffix}",
                    env=_build_env_from_ascii_path(
                        tmp_path, max_steps, border_width, num_agents
                    ),
                )
            )

    # Optional: simple combinations (rotation x mirror x stretch), single op from each
    if transform_combo:
        # Restrict to one from each selected family, generate Cartesian product
        pools: list[list[Transform]] = [families[f] for f in selected_families]
        for combo in product(*pools):
            # Apply in the standard order: rotation -> mirror -> stretch
            content = original_content
            suffix_parts: list[str] = []
            # Enforce order using family names
            by_family = {c.name: c for c in combo}
            if "rotate" in by_family:
                content = by_family["rotate"].apply(content)
                suffix_parts.append(by_family["rotate"].suffix)
            if "mirror" in by_family:
                content = by_family["mirror"].apply(content)
                suffix_parts.append(by_family["mirror"].suffix)
            if "stretch" in by_family:
                content = by_family["stretch"].apply(content)
                suffix_parts.append(by_family["stretch"].suffix)

            suffix = "_".join(suffix_parts)
            tmp_path = _save_transformed_map(content, original_ascii_map)
            sims.append(
                SimulationConfig(
                    name=f"{name}_{suffix}",
                    env=_build_env_from_ascii_path(
                        tmp_path, max_steps, border_width, num_agents
                    ),
                )
            )

    return sims


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

    # Helper to add a map with variants
    def transform_map(name: str, max_steps: int):
        sims.extend(
            apply_transformations(
                name,
                max_steps,
                transform_set="all",  # rotation + mirror + stretch
                transform_combo=False,  # only single transform per map
                include_original=True,
            )
        )

    transform_map("corridors", 450)
    transform_map("cylinder_easy", 250)
    transform_map("cylinder", 250)
    transform_map("honeypot", 300)
    transform_map("knotty", 500)
    transform_map("memory_palace", 200)
    transform_map("obstacles0", 100)
    transform_map("obstacles1", 300)
    transform_map("obstacles2", 350)
    transform_map("obstacles3", 300)
    transform_map("radial_large", 1000)
    transform_map("radial_mini", 150)
    transform_map("radial_small", 120)
    transform_map("radial_maze", 200)
    transform_map("swirls", 350)
    transform_map("thecube", 350)
    transform_map("walkaround", 250)
    transform_map("wanderout", 500)
    transform_map("emptyspace_outofsight", 150)
    transform_map("walls_outofsight", 250)
    transform_map("walls_withinsight", 120)
    transform_map("labyrinth", 250)

    # Non-ASCII procedural env left unchanged
    sims.append(
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env())
    )

    return sims
