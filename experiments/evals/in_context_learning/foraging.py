from experiments.recipes.in_context_learning.foraging import (
    make_foraging_env,
)
from metta.sim.simulation_config import SimulationConfig


def make_assembler_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/single_agent_two_altars_any",
            env=make_foraging_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                room_size="small",
                position=["Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/single_agent_two_altars_S",
            env=make_foraging_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                room_size="small",
                position=["S"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/single_agent_20_altars_any",
            env=make_foraging_env(
                num_agents=1,
                num_altars=24,
                num_generators=0,
                room_size="large",
                position=["Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/single_agent_many_altars_S",
            env=make_foraging_env(
                num_agents=1,
                num_altars=24,
                num_generators=0,
                room_size="large",
                position=["S"],
            ),
        ),
        SimulationConfig(
            name="in_context_learning_foraging/two_agent_two_altars_S_N",
            suite="in_context_learning_foraging",
            env=make_foraging_env(
                num_agents=2,
                num_altars=2,
                num_generators=0,
                room_size="small",
                position=["S", "N"],
            ),
        ),
        SimulationConfig(
            name="in_context_learning_foraging/12_agent_20_altars_N_S",
            suite="in_context_learning_foraging",
            env=make_foraging_env(
                num_agents=12,
                num_altars=20,
                num_generators=0,
                room_size="xlarge",
                position=["N", "S"],
            ),
        ),
        SimulationConfig(
            name="in_context_learning_foraging/12_agent_20_altars_any",
            suite="in_context_learning_foraging",
            env=make_foraging_env(
                num_agents=12,
                num_altars=20,
                num_generators=0,
                room_size="xlarge",
                position=["Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/large_3_agent_30_altars_any",
            env=make_foraging_env(
                num_agents=3,
                num_altars=30,
                room_size="xlarge",
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/3_agent_12_altars_any",
            env=make_foraging_env(
                num_agents=3,
                num_altars=12,
                room_size="large",
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/large_three_agent_30_altars_any",
            env=make_foraging_env(
                num_agents=12,
                num_altars=30,
                room_size="xlarge",
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/large_three_agent_many_altars_NSE",
            env=make_foraging_env(
                num_agents=3,
                num_altars=30,
                room_size="xlarge",
                num_generators=0,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="in_context_learning_foraging/large_12_agent_12_altars_NSE",
            env=make_foraging_env(
                num_agents=12,
                num_altars=30,
                room_size="xlarge",
                num_generators=0,
                position=["N", "S", "E"],
            ),
        ),
    ]
