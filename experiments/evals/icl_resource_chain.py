from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.icl_resource_chain import ConverterChainTaskGenerator


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
    chain_length: int, num_sinks: int
) -> MettaGridConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(0)


def make_icl_resource_chain_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="in_context_learning/chain_length2_1sink",
            env=make_icl_resource_chain_eval_env(2, 1),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length2_2sink",
            env=make_icl_resource_chain_eval_env(2, 2),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length3_0sink",
            env=make_icl_resource_chain_eval_env(3, 0),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length3_1sink",
            env=make_icl_resource_chain_eval_env(3, 1),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length3_2sink",
            env=make_icl_resource_chain_eval_env(3, 2),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length4_0sink",
            env=make_icl_resource_chain_eval_env(4, 0),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length4_1sink",
            env=make_icl_resource_chain_eval_env(4, 1),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length4_2sink",
            env=make_icl_resource_chain_eval_env(4, 2),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length5_0sink",
            env=make_icl_resource_chain_eval_env(5, 0),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length5_1sink",
            env=make_icl_resource_chain_eval_env(5, 1),
        ),
        SimulationConfig(
            name="in_context_learning/chain_length5_2sink",
            env=make_icl_resource_chain_eval_env(5, 2),
        ),
    ]
