import random

from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import FixedPosition, MettaGridConfig

from experiments.recipes.in_context_learning.assemblers.assembly_lines import (
    AssemblyLinesTaskGenerator,
    make_task_generator_cfg,
)


def make_icl_assembler_chain_eval_env(
    num_agents: int,
    chain_length: int,
    num_sinks: int,
    room_size: str,
    num_chests: int = 0,
    chest_position: list[FixedPosition] = ["N"],
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=[num_agents],
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        map_dir=None,
        num_chests=[num_chests],
        chest_positions=[chest_position],
    )
    task_generator = AssemblyLinesTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembly_line_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="single_agent_medium",
            env=make_icl_assembler_chain_eval_env(1, 3, 0, "medium"),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="single_agent_large",
            env=make_icl_assembler_chain_eval_env(1, 5, 2, "large"),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="two_agent_medium",
            env=make_icl_assembler_chain_eval_env(2, 3, 0, "medium"),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="two_agent_large",
            env=make_icl_assembler_chain_eval_env(2, 5, 2, "large"),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="three_agent_medium",
            env=make_icl_assembler_chain_eval_env(3, 3, 0, "medium"),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="three_agent_large",
            env=make_icl_assembler_chain_eval_env(3, 5, 2, "large"),
        ),
    ]
