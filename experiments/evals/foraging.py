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
    # Back-compat single-value overrides
    width: Optional[int] = None,
    height: Optional[int] = None,
    deposit_count: Optional[int] = None,
    deposits_per_resource: Optional[Dict[str, int]] = None,
    # New: per-sim variations
    widths: Optional[List[int]] = None,
    heights: Optional[List[int]] = None,
    deposit_counts: Optional[List[int]] = None,
) -> List[SimulationConfig]:
    sims: list[SimulationConfig] = []

    width_choices: List[Optional[int]] = list(widths) if widths is not None else [width]
    height_choices: List[Optional[int]] = (
        list(heights) if heights is not None else [height]
    )
    dep_choices: List[Optional[int]] = (
        list(deposit_counts) if deposit_counts is not None else [deposit_count]
    )

    for num_converters in [2, 4]:
        for carry in [1]:
            for cooldowns in ([0], [10]):
                for rmin, rmax in [(2, 3), (3, 10)]:
                    for w in width_choices:
                        for h in height_choices:
                            for dep in dep_choices:
                                dims = (
                                    f"_{w}x{h}"
                                    if (w is not None and h is not None)
                                    else ""
                                )
                                dep_suffix = (
                                    f"_dep{dep}"
                                    if dep is not None
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
                                            width=w,
                                            height=h,
                                            deposits_per_resource=deposits_per_resource,
                                            deposit_count=dep,
                                        ),
                                    )
                                )
    return sims


def make_foraging_eval_pair() -> List[SimulationConfig]:
    sims: list[SimulationConfig] = []

    # Map 1: 2 converters, rmin=2, rmax=5, size 16x10, deposit_count=2
    sims.append(
        SimulationConfig(
            name="foraging/two_maps/n2_r2-5_16x10_dep2",
            env=make_foraging_eval_env(
                num_converters=2,
                carry=1,
                cooldowns=[0],
                rmin=2,
                rmax=5,
                width=16,
                height=10,
                deposit_count=2,
            ),
        )
    )

    # Map 2: 4 converters, rmin=5, rmax=10, size 40x30, deposit_count=5
    sims.append(
        SimulationConfig(
            name="foraging/two_maps/n4_r5-10_40x30_dep5",
            env=make_foraging_eval_env(
                num_converters=4,
                carry=1,
                cooldowns=[0],
                rmin=5,
                rmax=10,
                width=40,
                height=30,
                deposit_count=5,
            ),
        )
    )

    return sims
