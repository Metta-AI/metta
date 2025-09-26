from experiments.recipes.in_context_learning.assemblers import make_assembler_env
from metta.sim.simulation_config import SimulationConfig


def make_assembler_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/single_agent_two_altars_W",
            env=make_assembler_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                width=8,
                height=8,
                position=["W"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/two_agent_two_altars_any",
            env=make_assembler_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                width=8,
                height=8,
                position=["Any"],
            ),
        ),
        SimulationConfig(
            name="in_context_learning_assemblers/two_agent_two_altars_west_east",
            suite="in_context_learning_assemblers",
            env=make_assembler_env(
                num_agents=2,
                num_altars=2,
                num_generators=0,
                width=8,
                height=8,
                position=["W", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/large_three_agent_many_altars_any",
            env=make_assembler_env(
                num_agents=3,
                num_altars=30,
                width=30,
                height=30,
                num_generators=0,
                position=["Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/large_three_agent_many_altars_NSE",
            env=make_assembler_env(
                num_agents=3,
                num_altars=30,
                width=30,
                height=30,
                num_generators=0,
                position=["N", "S", "E"],
            ),
        ),
    ]
