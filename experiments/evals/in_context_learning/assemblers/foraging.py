from experiments.recipes.in_context_learning.assemblers.foraging import (
    make_assembler_env,
)
from metta.sim.simulation_config import SimulationConfig


def make_foraging_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="single_agent_two_altars_S",
            env=make_assembler_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                room_size="small",
                position=["S"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="single_agent_many_altars_S",
            env=make_assembler_env(
                num_agents=1,
                num_altars=20,
                num_generators=0,
                room_size="large",
                position=["S"],
            ),
        ),
        SimulationConfig(
            name="two_agent_two_altars_S_N",
            suite="in_context_learning_foraging",
            env=make_assembler_env(
                num_agents=2,
                num_altars=2,
                num_generators=0,
                room_size="small",
                position=["S", "N"],
            ),
        ),
        SimulationConfig(
            name="12_agent_20_altars_N_S",
            suite="in_context_learning_foraging",
            env=make_assembler_env(
                num_agents=12,
                num_altars=20,
                num_generators=0,
                room_size="xlarge",
                position=["N", "S"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="3_agent_12_altars_any",
            env=make_assembler_env(
                num_agents=3,
                num_altars=12,
                room_size="large",
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="xlarge_three_agent_20_altars_NSE",
            env=make_assembler_env(
                num_agents=3,
                num_altars=20,
                room_size="xlarge",
                num_generators=0,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="three_agent_1_altar_1_generator",
            env=make_assembler_env(
                num_agents=3,
                num_altars=1,
                room_size="large",
                num_generators=1,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="three_agent_5_altar_5_generator",
            env=make_assembler_env(
                num_agents=3,
                num_altars=5,
                room_size="large",
                num_generators=5,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="12_agent_5_altar_5_generator_NS",
            env=make_assembler_env(
                num_agents=12,
                num_altars=5,
                room_size="xlarge",
                num_generators=5,
                position=["N", "S"],
            ),
        ),
    ]
