from experiments.recipes.cogs_v_clips.foraging import (
    make_env,
)
from metta.sim.simulation_config import SimulationConfig


def make_foraging_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="two_agent_two_assembler_S_N",
            suite="in_context_learning_foraging",
            env=make_env(
                num_cogs=2,
                num_assemblers=2,
                num_extractors=0,
                sizes="small",
                position=["S", "N"],
            ),
        ),
        SimulationConfig(
            name="12_agent_20_assemblers_N_S",
            suite="in_context_learning_foraging",
            env=make_env(
                num_cogs=12,
                num_assemblers=20,
                num_extractors=0,
                sizes="xlarge",
                position=["N", "S"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="3_agent_12_assemblers_any",
            env=make_env(
                num_cogs=3,
                num_assemblers=12,
                sizes="large",
                num_extractors=0,
                position=["Any", "Any", "Any"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="xlarge_three_agent_20_assemblers_NSE",
            env=make_env(
                num_cogs=3,
                num_assemblers=20,
                sizes="xlarge",
                num_extractors=0,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="three_agent_1_assembler_1_extractor",
            env=make_env(
                num_cogs=3,
                num_assemblers=1,
                sizes="large",
                num_extractors=1,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="three_agent_5_assembler_5_extractor",
            env=make_env(
                num_cogs=3,
                num_assemblers=5,
                sizes="large",
                num_extractors=5,
                position=["N", "S", "E"],
            ),
        ),
        SimulationConfig(
            suite="in_context_learning_foraging",
            name="12_agent_5_assembler_5_extractor_NS",
            env=make_env(
                num_cogs=12,
                num_assemblers=5,
                sizes="xlarge",
                num_extractors=5,
                position=["N", "S"],
            ),
        ),
    ]
