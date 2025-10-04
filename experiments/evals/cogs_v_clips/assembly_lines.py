from experiments.recipes.cogs_v_clips.assembly_lines import (
    make_env,
)
from metta.sim.simulation_config import SimulationConfig


def make_assembly_lines_eval_suite():
    return [
        SimulationConfig(
            name="chain_1_NS",
            suite="assembly_lines",
            env=make_env(num_cogs=2, chain_length=1, position=["N", "S"]),
        ),
        SimulationConfig(
            name="chain_2_NS",
            suite="assembly_lines",
            env=make_env(num_cogs=2, chain_length=2, position=["N", "S"]),
        ),
        SimulationConfig(
            name="chain_3_NS",
            suite="assembly_lines",
            env=make_env(num_cogs=2, chain_length=3, position=["N", "S"]),
        ),
        SimulationConfig(
            name="chain_4_NS",
            suite="assembly_lines",
            env=make_env(num_cogs=2, chain_length=4, position=["N", "S"]),
        ),
        SimulationConfig(
            name="chain_5_NS",
            suite="assembly_lines",
            env=make_env(num_cogs=2, chain_length=5, position=["N", "S"]),
        ),
        SimulationConfig(
            name="chain_1_NSE",
            suite="assembly_lines",
            env=make_env(num_cogs=3, chain_length=1, position=["N", "S", "E"]),
        ),
        SimulationConfig(
            name="chain_2_NSE",
            suite="assembly_lines",
            env=make_env(num_cogs=3, chain_length=2, position=["N", "S", "E"]),
        ),
        SimulationConfig(
            name="chain_3_NSE",
            suite="assembly_lines",
            env=make_env(num_cogs=3, chain_length=3, position=["N", "S", "E"]),
        ),
        SimulationConfig(
            name="chain_4_NSE",
            suite="assembly_lines",
            env=make_env(num_cogs=3, chain_length=4, position=["N", "S", "E"]),
        ),
        SimulationConfig(
            name="chain_5_NSE",
            suite="assembly_lines",
            env=make_env(num_cogs=3, chain_length=5, position=["N", "S", "E"]),
        ),
    ]
