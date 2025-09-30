import random

from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig, Position

from experiments.recipes.in_context_learning.assembly_lines import (
    AssemblyLinesTaskGenerator,
    make_task_generator_cfg,
)


def make_icl_assembler_chain_eval_env(
    num_agents: int,
    chain_length: int,
    num_sinks: int,
    room_size: str,
    positions: list[list[Position]],
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=[num_agents],
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        positions=positions,
        map_dir=None,
    )
    task_generator = AssemblyLinesTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembly_line_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="single_agent_medium",
            env=make_icl_assembler_chain_eval_env(1, 3, 0, "medium", [["Any"]]),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="single_agent_large",
            env=make_icl_assembler_chain_eval_env(1, 5, 2, "large", [["Any"]]),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="two_agent_medium",
            env=make_icl_assembler_chain_eval_env(2, 3, 0, "medium", [["Any", "Any"]]),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="two_agent_large",
            env=make_icl_assembler_chain_eval_env(2, 5, 2, "large", [["Any", "Any"]]),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="three_agent_medium",
            env=make_icl_assembler_chain_eval_env(
                3, 3, 0, "medium", [["Any", "Any", "Any"]]
            ),
        ),
        SimulationConfig(
            suite="in_context_assembly_lines",
            name="three_agent_large",
            env=make_icl_assembler_chain_eval_env(
                3, 5, 2, "large", [["Any", "Any", "Any"]]
            ),
        ),
    ]
