from __future__ import annotations

from typing import Dict, List, Optional

from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.foraging import ForagingTaskGenerator


def make_foraging_eval_env(
    num_converters: int,
    carry: int,
    cooldowns: list[int],
    rmin: int,
    rmax: int,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    deposits_per_resource: Optional[Dict[str, int]] = None,
    deposit_count: Optional[int] = None,
) -> MettaGridConfig:
    cfg = ForagingTaskGenerator.Config(
        num_converters=num_converters,
        default_resource_limit=carry,
        cooldowns=cooldowns,
        outside_min_radius=rmin,
        outside_max_radius=rmax,
    )

    if width is not None:
        cfg.width = int(width)
    if height is not None:
        cfg.height = int(height)

    if deposits_per_resource is not None:
        cfg.deposits_per_resource = {
            str(k): int(v) for k, v in deposits_per_resource.items()
        }
    elif deposit_count is not None:
        # Apply a uniform count to common ore resources
        cfg.deposits_per_resource = {
            "ore_red": int(deposit_count),
            "ore_blue": int(deposit_count),
            "ore_green": int(deposit_count),
        }

    gen = ForagingTaskGenerator(cfg)
    return gen.get_task(0)


def make_foraging_eval_suite(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    deposits_per_resource: Optional[Dict[str, int]] = None,
    deposit_count: Optional[int] = None,
) -> List[SimulationConfig]:
    sims: list[SimulationConfig] = []
    for num_converters in [2, 3, 4]:
        for carry in [1, 2]:
            for cooldowns in ([0], [5], [10], [0, 5, 10]):
                for rmin, rmax in [(2, 3), (3, 6)]:
                    dims = f"_{width}x{height}" if width and height else ""
                    dep_suffix = (
                        f"_dep{deposit_count}"
                        if deposit_count is not None
                        else ("_depmap" if deposits_per_resource else "")
                    )
                    name = f"foraging/n{num_converters}_carry{carry}_cd{'-'.join(map(str, cooldowns))}_r{rmin}-{rmax}{dims}{dep_suffix}"
                    sims.append(
                        SimulationConfig(
                            name=name,
                            env=make_foraging_eval_env(
                                num_converters,
                                carry,
                                list(cooldowns),
                                rmin,
                                rmax,
                                width=width,
                                height=height,
                                deposits_per_resource=deposits_per_resource,
                                deposit_count=deposit_count,
                            ),
                        )
                    )
    return sims
