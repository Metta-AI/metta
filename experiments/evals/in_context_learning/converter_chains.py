from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
import random
from experiments.recipes.in_context_learning.converters.converter_chains import (
    ConverterChainTaskGenerator,
    make_task_generator_cfg,
)


def converter_chain_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    env.game.agent.resource_limits["heart"] = 6
    return env


def make_converter_chain_eval_env(
    chain_length: int,
    num_sinks: int,
    room_size: str,
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        map_dir=None,  # for evals, generate the environments algorithmically
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_converter_chain_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium",
            env=make_converter_chain_eval_env(2, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium_terrain",
            env=make_converter_chain_eval_env(2, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_1s_medium",
            env=make_converter_chain_eval_env(3, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_2s_medium",
            env=make_converter_chain_eval_env(3, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain_dense",
            env=make_converter_chain_eval_env(4, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain",
            env=make_converter_chain_eval_env(4, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_medium_terrain",
            env=make_converter_chain_eval_env(4, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_large_terrain",
            env=make_converter_chain_eval_env(4, 2, "large"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium",
            env=make_converter_chain_eval_env(5, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium_terrain",
            env=make_converter_chain_eval_env(5, 1, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_medium_terrain",
            env=make_converter_chain_eval_env(5, 2, "medium"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_large_terrain_dense",
            env=make_converter_chain_eval_env(5, 2, "large"),
        ),
    ]
