from softmax.training.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig

from experiments.recipes.in_context_learning.icl_resource_chain import ICLTaskGenerator
from experiments.recipes.in_context_learning.unordered_chains import (
    UnorderedChainTaskGenerator,
)


def icl_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    # Remove heart limit for evals to focus on recipe complexity
    env.game.agent.resource_limits["heart"] = 255
    return env


def make_unordered_chain_eval_env(
    num_resources: int,
    num_converters: int,
    room_size: str = "small",
    max_recipe_inputs: int | None = None,
    singleton_resources: bool = False,
) -> MettaGridConfig:
    # Unordered chain uses explicit converter count
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[num_resources],
        num_converters=[num_converters],
        room_sizes=[room_size],
        max_recipe_inputs=[max_recipe_inputs]
        if max_recipe_inputs is not None
        else None,
        # Configure singleton behavior: 1 resource per source, no regeneration
        source_initial_resource_count=1 if singleton_resources else None,
        source_max_conversions=0 if singleton_resources else None,
    )
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_unordered_chain_eval_suite() -> list[SimulationConfig]:
    """Test recipe learning and scavenging."""
    combos = [
        # Recipe Learning (continuous resources)
        (4, 1, "small", 1, False),
        (4, 2, "small", 2, False),
        (4, 3, "small", 3, False),
        # Scavenging (singleton resources)
        (4, 1, "medium", 1, True),
        (4, 2, "medium", 2, True),
        (4, 3, "large", 3, True),
    ]

    sims: list[SimulationConfig] = []
    for num_resources, num_converters, room_size, max_inputs, singleton in combos:
        res_type = "singleton" if singleton else "continuous"
        sim_name = f"unordered/{res_type}_r{num_resources}_c{num_converters}_i{max_inputs}_{room_size}"
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
                        singleton_resources=singleton,
                    )
                ),
            )
        )
    return sims
