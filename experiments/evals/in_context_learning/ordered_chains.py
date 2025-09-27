from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
import random
from experiments.recipes.in_context_learning.ordered_chains import (
    OrderedChainsTaskGenerator,
    make_task_generator_cfg,
)


def icl_resource_chain_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    env.game.agent.resource_limits["heart"] = 6
    return env


def update_recipe(converter, input_resource=None, output_resource=None):
    if input_resource is not None:
        converter.input_resources = {input_resource: 1}
    if output_resource is not None:
        converter.output_resources = {output_resource: 1}
    return converter


def make_icl_resource_chain_eval_env(
    chain_length: int,
    num_sinks: int,
    room_size: str,
    obstacle_types: list[str] = [],
    densities: list[str] = [],
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        obstacle_types=obstacle_types,
        densities=densities,
        map_dir=None,  # for evals, generate the environments algorithmically
    )
    task_generator = OrderedChainsTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_icl_resource_chain_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium",
            env=make_icl_resource_chain_eval_env(2, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium_terrain",
            env=make_icl_resource_chain_eval_env(
                2, 2, "medium", ["square"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_1s_medium",
            env=make_icl_resource_chain_eval_env(3, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_2s_medium",
            env=make_icl_resource_chain_eval_env(3, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain_dense",
            env=make_icl_resource_chain_eval_env(4, 1, "medium", ["cross"], ["dense"]),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain",
            env=make_icl_resource_chain_eval_env(
                4, 1, "medium", ["cross"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_medium_terrain",
            env=make_icl_resource_chain_eval_env(
                4, 2, "medium", ["cross"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_large_terrain",
            env=make_icl_resource_chain_eval_env(
                4, 2, "large", ["cross"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium",
            env=make_icl_resource_chain_eval_env(5, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium_terrain",
            env=make_icl_resource_chain_eval_env(
                5, 1, "medium", ["cross"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_medium_terrain",
            env=make_icl_resource_chain_eval_env(
                5, 2, "medium", ["cross"], ["balanced"]
            ),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_large_terrain_dense",
            env=make_icl_resource_chain_eval_env(5, 2, "large", ["cross"], ["dense"]),
        ),
    ]
