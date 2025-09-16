from __future__ import annotations

from typing import List

from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.foraging import ForagingTaskGenerator


def make_foraging_eval_env(
    num_converters: int,
    carry: int,
    cooldowns: list[int],
    rmin: int,
    rmax: int,
) -> MettaGridConfig:
    cfg = ForagingTaskGenerator.Config(
        num_converters=num_converters,
        default_resource_limit=carry,
        cooldowns=cooldowns,
        outside_min_radius=rmin,
        outside_max_radius=rmax,
    )
    gen = ForagingTaskGenerator(cfg)
    return gen.get_task(0)


def make_foraging_eval_suite() -> List[SimulationConfig]:
    sims: list[SimulationConfig] = []
    for num_converters in [2, 10]:
        for carry in [1]:
            for cooldowns in ([0], [10]):
                for rmin, rmax in [(2, 2), (6, 6)]:
                    name = f"foraging/n{num_converters}_carry{carry}_cd{'-'.join(map(str, cooldowns))}_r{rmin}-{rmax}"
                    sims.append(
                        SimulationConfig(
                            name=name,
                            env=make_foraging_eval_env(
                                num_converters, carry, list(cooldowns), rmin, rmax
                            ),
                        )
                    )
    return sims
