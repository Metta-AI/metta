import random

from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig

from experiments.recipes.machina1.single_agent_base import (
    Machina1BaseTaskGenerator,
    make_task_generator_cfg,
)


def make_eval_env(
    extractors, use_charger, use_chest, use_extractor_glyphs, efficiences, max_uses
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        extractors=extractors,
        use_charger=use_charger,
        use_chest=use_chest,
        use_extractor_glyphs=use_extractor_glyphs,
        efficiences=efficiences,
        max_uses=max_uses,
    )
    task_generator = Machina1BaseTaskGenerator(task_generator_cfg)
    # different set of resources and converters for evals
    return task_generator.get_task(random.randint(0, 1000000))


def make_single_agent_base_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="single_agent_base",
            name="assembler_and_chest",
            env=make_eval_env(
                extractors=[],
                use_charger=False,
                use_chest=True,
                use_extractor_glyphs=False,
                efficiences=100,
                max_uses=1000,
            ),
        ),
        SimulationConfig(
            suite="single_agent_base",
            name="one_extractor",
            env=make_eval_env(
                extractors=["carbon"],
                use_charger=False,
                use_chest=True,
                use_extractor_glyphs=False,
                efficiences=100,
                max_uses=1000,
            ),
        ),
        SimulationConfig(
            suite="single_agent_base",
            name="two_extractors",
            env=make_eval_env(
                extractors=["carbon", "oxygen"],
                use_charger=False,
                use_chest=True,
                use_extractor_glyphs=False,
                efficiences=100,
                max_uses=1000,
            ),
        ),
        SimulationConfig(
            suite="single_agent_base",
            name="three_extractors",
            env=make_eval_env(
                extractors=["carbon", "oxygen", "germanium"],
                use_charger=False,
                use_chest=True,
                use_extractor_glyphs=False,
                efficiences=100,
                max_uses=1000,
            ),
        ),
        SimulationConfig(
            suite="single_agent_base",
            name="four_extractors",
            env=make_eval_env(
                extractors=["carbon", "oxygen", "germanium", "silicon"],
                use_charger=False,
                use_chest=True,
                use_extractor_glyphs=False,
                efficiences=100,
                max_uses=1000,
            ),
        ),
    ]
