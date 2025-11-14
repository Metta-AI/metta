#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH = REPO_ROOT / "devops" / "skypilot" / "launch.py"


@dataclass(frozen=True)
class Variant:
    module: str
    base_overrides: Sequence[str] = field(default_factory=list)
    extra_overrides: Callable[[int, str, str], Sequence[str]] = field(default_factory=lambda: (lambda *_: []))


def _navigation_overrides(_: int, map_size: str, terrain: str) -> list[str]:
    map_dir = f"varied_terrain/{terrain}_{map_size}"
    bucket_key = "training_env.curriculum.task_generator.task_generators.0.buckets.game.map_builder.instance.dir"
    return [f"{bucket_key}={json.dumps([map_dir])}"]


VARIANTS: dict[str, Variant] = {
    "arena": Variant(
        module="arena.train",
        base_overrides=[
            "training_env.curriculum.algorithm_config.exploration_bonus=0.1",
        ],
    ),
    "arena_sparse": Variant(
        module="arena_with_sparse_rewards.train",
        base_overrides=[
            "training_env.curriculum.algorithm_config.exploration_bonus=0.15",
        ],
    ),
    "navigation": Variant(
        module="navigation.train",
        base_overrides=[
            "training_env.curriculum.algorithm_config.max_slice_axes=4",
        ],
        extra_overrides=_navigation_overrides,
    ),
}

SEEDS = [0, 1, 2]
MAP_SIZES = ["small", "medium", "large"]
MAP_TERRAINS = ["balanced", "dense", "sparse", "maze", "cylinder-world"]
GPUS_PER_NODE = 8


def main() -> None:
    for (variant_name, variant), seed, map_size, terrain in itertools.product(
        VARIANTS.items(),
        SEEDS,
        MAP_SIZES,
        MAP_TERRAINS,
    ):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = f"{variant_name}_seed{seed}_{map_size}_{terrain}_{timestamp}"

        overrides = [
            f"training_env.seed={seed}",
            *variant.base_overrides,
            *variant.extra_overrides(seed, map_size, terrain),
        ]

        cmd = [
            str(LAUNCH),
            variant.module,
            "--gpus",
            str(GPUS_PER_NODE),
            "--",
            f"run={run_id}",
            *overrides,
        ]

        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
