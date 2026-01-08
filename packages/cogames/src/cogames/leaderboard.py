from __future__ import annotations

from typing import Optional

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from mettagrid import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen


def make_machina1_open_world_env(
    *, num_cogs: int, seed: Optional[int] = None, map_seed: Optional[int] = None, steps: Optional[int] = None
) -> MettaGridConfig:
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env_cfg = mission.make_env()
    if steps is not None:
        env_cfg.game.max_steps = steps

    effective_map_seed: Optional[int]
    if map_seed is not None:
        effective_map_seed = map_seed
    else:
        effective_map_seed = seed

    if effective_map_seed is not None:
        map_builder = getattr(env_cfg.game, "map_builder", None)
        if isinstance(map_builder, MapGen.Config):
            map_builder.seed = effective_map_seed

    return env_cfg


def allocate_counts(total: int, weights: list[float]) -> list[int]:
    total_weight = sum(weights)
    fractions = [weight / total_weight for weight in weights]
    ideals = [total * fraction for fraction in fractions]
    counts = [int(value) for value in ideals]
    remaining = total - sum(counts)

    remainders = [ideal - count for ideal, count in zip(ideals, counts, strict=True)]
    for i in sorted(range(len(remainders)), key=remainders.__getitem__, reverse=True)[:remaining]:
        counts[i] += 1
    return counts
