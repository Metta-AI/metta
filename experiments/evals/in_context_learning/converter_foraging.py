from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig

from experiments.recipes.in_context_learning.in_context_learning import ICLTaskGenerator
from experiments.recipes.in_context_learning.converters.converter_foraging import (
    ConverterForagingTaskGenerator,
)


def icl_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    # Remove heart limit for evals to focus on recipe complexity
    env.game.agent.resource_limits["heart"] = 255
    return env


def make_unordered_chain_eval_env(
    num_resources: int,
    num_converters: int,
    max_recipe_inputs: int,
    room_size: str = "small",
) -> MettaGridConfig:
    # Unordered chain uses explicit converter count
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[num_resources],
        num_converters=[num_converters],
        room_sizes=[room_size],
        max_recipe_inputs=[max_recipe_inputs],
    )
    task_generator = ConverterForagingTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_unordered_chain_eval_suite() -> list[SimulationConfig]:
    """Test recipe learning and scavenging."""
    combos = [
        # Recipe Learning (continuous resources)
        (4, 1, "small", 1),
        (4, 2, "small", 2),
        (4, 3, "small", 3),
        # Scavenging (singleton resources)
        (4, 1, "medium", 1),
        (4, 2, "medium", 2),
        (4, 3, "large", 3),
    ]

    sims: list[SimulationConfig] = []
    for num_resources, num_converters, room_size, max_inputs in combos:
        sim_name = (
            f"unordered/r{num_resources}_c{num_converters}_i{max_inputs}_{room_size}"
        )
        sims.append(
            SimulationConfig(
                suite="in_context_learning",
                name=sim_name,
                env=icl_eval_env(
                    make_unordered_chain_eval_env(
                        num_resources=num_resources,
                        num_converters=num_converters,
                        room_size=room_size,
                        max_recipe_inputs=max_inputs,
                    )
                ),
            )
        )
    return sims
