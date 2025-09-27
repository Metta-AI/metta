from experiments.recipes.in_context_learning.assemblers import make_assembler_env
from metta.sim.simulation_config import SimulationConfig


def make_assembler_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/single_agent_two_altars_any",
            env=make_assembler_env(
                num_agents=1,
                num_altars = 2,
                num_generators=0,
                width=8,
                height=8,
                position=["Any"],
        ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/single_agent_two_altars_S",
            env=make_assembler_env(
                num_agents=1,
                num_altars = 2,
                num_generators=0,
                width=8,
                height=8,
                position=["S"],
        ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/single_agent_20_altars_any",
            env=make_assembler_env(
                num_agents=1,
                num_altars = 24,
                num_generators=0,
                width=20,
                height=20,
                position=["Any"],
        ),
        ),
        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/single_agent_many_altars_S",
            env=make_assembler_env(
                num_agents=1,
                num_altars = 24,
                num_generators=0,
                width=20,
                height=20,
                position=["S"],
        ),
        ),
        # SimulationConfig(
        #     suite="in_context_learning_assemblers",
        #     name="in_context_learning_assemblers/single_agent_many_altars_sparse_any",
        #     env=make_assembler_env(
        #         num_agents=1,
        #         num_altars = 24,
        #         num_generators=0,
        #         width=60,
        #         height=60,
        #         position=["Any"],
        # ),
        # ),
        # SimulationConfig(
        #     suite="in_context_learning_assemblers",
        #     name="in_context_learning_assemblers/single_agent_many_altars_sparse_S",
        #     env=make_assembler_env(
        #         num_agents=1,
        #         num_altars = 24,
        #         num_generators=0,
        #         width=60,
        #         height=60,
        #         position=["S"],
        # ),
        # ),

        SimulationConfig(
            name="in_context_learning_assemblers/two_agent_two_altars_S_N",
            suite="in_context_learning_assemblers",
            env=make_assembler_env(
                num_agents=2,
                num_altars=2,
                num_generators=0,
                width=8,
                height=8,
                position=["S", "N"],
            ),
        ),

        SimulationConfig(
            name="in_context_learning_assemblers/12_agent_20_altars_N_S",
            suite="in_context_learning_assemblers",
            env=make_assembler_env(
                num_agents=12,
                num_altars=20,
                num_generators=0,
                width=40,
                height=40,
                position=["N", "S"],
            ),
        ),

        SimulationConfig(
            name="in_context_learning_assemblers/12_agent_20_altars_any",
            suite="in_context_learning_assemblers",
            env=make_assembler_env(
                num_agents=12,
                num_altars=20,
                num_generators=0,
                width=40,
                height=40,
                position=["Any", "Any"],
            ),
        ),

        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/large_3_agent_30_altars_any",
            env=make_assembler_env(
                num_agents=3,
                num_altars=30,
                width=30,
                height=30,
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),

        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/3_agent_12_altars_any",
            env=make_assembler_env(
                num_agents=3,
                num_altars=12,
                width=15,
                height=15,
                num_generators=0,
                position=["Any", "Any", "Any"],
            ),
        ),

        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/large_three_agent_30_altars_any",
            env=make_assembler_env(
                num_agents=12,
                num_altars=30,
                width=30,
                height=30,
                num_generators=0,
                position=["Any", "Any", "Any"],
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

        SimulationConfig(
            suite="in_context_learning_assemblers",
            name="in_context_learning_assemblers/large_12_agent_12_altars_NSE",
            env=make_assembler_env(
                num_agents=12,
                num_altars=30,
                width=50,
                height=50,
                num_generators=0,
                position=["N", "S", "E"],
            ),
        ),
    ]
