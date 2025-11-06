import random

import metta.sim.simulation_config
import mettagrid.config.mettagrid_config

import experiments.recipes.assembly_lines


def make_assembly_line_eval_env(
    chain_length: int,
    num_sinks: int,
    room_size: str,
    terrain: str,
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    task_generator_cfg = experiments.recipes.assembly_lines.make_task_generator_cfg(
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        terrains=[terrain],
    )
    task_generator = experiments.recipes.assembly_lines.AssemblyLinesTaskGenerator(
        task_generator_cfg
    )
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembly_line_eval_suite() -> list[
    metta.sim.simulation_config.SimulationConfig
]:
    return [
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium",
            env=make_assembly_line_eval_env(2, 2, "medium", "no-terrain"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium_balanced",
            env=make_assembly_line_eval_env(2, 2, "medium", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_1s_medium",
            env=make_assembly_line_eval_env(3, 1, "medium", "no-terrain"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_2s_medium",
            env=make_assembly_line_eval_env(3, 2, "medium", "no-terrain"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain_dense",
            env=make_assembly_line_eval_env(4, 1, "medium", "dense"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_balanced",
            env=make_assembly_line_eval_env(4, 1, "medium", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_medium_balanced",
            env=make_assembly_line_eval_env(4, 2, "medium", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_large_balanced",
            env=make_assembly_line_eval_env(4, 2, "large", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium",
            env=make_assembly_line_eval_env(5, 1, "medium", "no-terrain"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium_balanced",
            env=make_assembly_line_eval_env(5, 1, "medium", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_medium_balanced",
            env=make_assembly_line_eval_env(5, 2, "medium", "balanced"),
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_large_dense",
            env=make_assembly_line_eval_env(5, 2, "large", "dense"),
        ),
    ]
