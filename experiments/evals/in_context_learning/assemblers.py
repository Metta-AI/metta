from experiments.recipes.in_context_learning.assemblers import make_assembler_env
from metta.sim.simulation_config import SimulationConfig


def make_assembler_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="in_context_assemblers/single_agent_two_altars_W",
            suite="in_context_learning",
            env=make_assembler_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                width=6,
                height=6,
                position=["W"],
            ),
        ),
        SimulationConfig(
            name="in_context_assemblers/two_agent_two_altars_any",
            suite="in_context_learning",
            env=make_assembler_env(
                num_agents=2,
                num_altars=2,
                num_generators=0,
                width=6,
                height=6,
                position=["Any"],
            ),
        ),
        SimulationConfig(
            name="in_context_assemblers/two_agent_two_altars_north_south",
            suite="in_context_learning",
            env=make_assembler_env(
                num_agents=1,
                num_altars=2,
                num_generators=0,
                width=6,
                height=6,
                position=["N", "S"],
            ),
        ),
    ]
