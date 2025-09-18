from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.in_context_learning.unordered_chain import (
    ICLTaskGenerator,
    UnorderedChainTaskGenerator,
)


def icl_unordered_chain_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    # Remove heart limit for evals to focus on recipe complexity
    env.game.agent.resource_limits["heart"] = 9999  # Effectively unlimited
    return env


def make_icl_unordered_chain_eval_env(
    num_resources: int,
    num_converters: int,
    room_size: str = "small",
    max_recipe_inputs: int | None = None,
    singleton_resources: bool = False,
) -> MettaGridConfig:
    # Unordered chain uses ICLTaskGenerator.Config; `num_sinks` acts as converter count
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[num_resources],
        num_sinks=[num_converters],
        room_sizes=[room_size],
        max_recipe_inputs=[max_recipe_inputs]
        if max_recipe_inputs is not None
        else None,
        # Configure singleton behavior: 1 resource per source, no regeneration
        source_initial_resource_count=1 if singleton_resources else None,
        source_max_conversions=0 if singleton_resources else None,
        source_cooldown=25,
    )
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_icl_unordered_chain_eval_suite() -> list[SimulationConfig]:
    """Test scenarios for two key behaviors:

    1. Recipe Learning: Can agents learn and repeat new recipes? (continuous resources)
    2. Scavenging: Can agents efficiently search, gather, and remember resource locations? (singleton resources)

    Each behavior is tested with varying:
    - Number of unique recipes (num_converters)
    - Recipe complexity (max_recipe_inputs)

    Tuple format: (num_resources, num_converters, room_size, max_recipe_inputs, singleton)
    """
    combos: list[tuple[int, int, str, int, bool]] = [
        # Recipe Learning Tests (continuous resources, small maps)
        # Focus: Learn recipes without navigation challenges
        (4, 1, "small", 1, False),  # 1 simple recipe
        (4, 1, "small", 2, False),  # 1 medium recipe
        (4, 2, "small", 1, False),  # 2 simple recipes
        (4, 2, "small", 2, False),  # 2 medium recipes
        (4, 3, "small", 2, False),  # 3 medium recipes (cap resources to 4)
        (4, 3, "small", 3, False),  # 3 complex recipes (cap resources to 4)
        # Scavenging Tests (singleton resources, larger maps)
        # Focus: Efficient exploration and resource gathering
        (4, 1, "medium", 1, True),  # Simple recipe, focus on navigation
        (4, 1, "medium", 2, True),  # Medium recipe, resource collection
        (4, 2, "medium", 2, True),  # 2 recipes, resource management
        (4, 2, "large", 3, True),  # Complex recipes, large search space
        (4, 3, "large", 2, True),  # Many sources to find
        (4, 3, "large", 3, True),  # Maximum challenge
    ]

    sims: list[SimulationConfig] = []
    for (
        num_resources,
        num_converters,
        room_size,
        max_recipe_inputs,
        singleton,
    ) in combos:
        resource_type = "singleton" if singleton else "continuous"
        sims.append(
            SimulationConfig(
                name=f"in_context_learning_unordered/{resource_type}_r{num_converters}_c{max_recipe_inputs}_{room_size}",
                env=icl_unordered_chain_eval_env(
                    make_icl_unordered_chain_eval_env(
                        num_resources=num_resources,
                        num_converters=num_converters,
                        room_size=room_size,
                        max_recipe_inputs=max_recipe_inputs,
                        singleton_resources=singleton,
                    )
                ),
            )
        )
    return sims
