from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.in_context_learning.unordered_chain import (
    ICLTaskGenerator,
    UnorderedChainTaskGenerator,
)


def icl_unordered_chain_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    # Optionally tweak agent limits/rewards for evals
    env.game.agent.resource_limits["heart"] = max(
        6, env.game.agent.resource_limits.get("heart", 0)
    )
    return env


def make_icl_unordered_chain_eval_env(
    num_resources: int, num_converters: int, room_size: str = "small"
) -> MettaGridConfig:
    # Unordered chain uses ICLTaskGenerator.Config; `num_sinks` acts as converter count
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[num_resources],
        num_sinks=[num_converters],
        room_sizes=[room_size],
    )
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_icl_unordered_chain_eval_suite() -> list[SimulationConfig]:
    """Test scenarios:
    - Single source with single converter (basic recipe learning)
    - Multiple sources with single converter (resource selection)
    - Multiple sources with multiple converters (complex recipes)
    - Many sources with many converters (rich environment)
    """
    combos: list[tuple[int, int, str]] = [
        # Basic: 1-2 resources, 1 converter
        (1, 1, "small"),
        (2, 1, "small"),
        # Medium complexity: 2-3 resources, 2-3 converters
        (2, 2, "small"),
        (3, 2, "small"),
        (3, 3, "medium"),
        # Complex: 4-5 resources, multiple converters
        (4, 2, "medium"),
        (4, 3, "medium"),
        (5, 3, "medium"),
        # Rich environments: many resources and converters
        (5, 5, "large"),
        (6, 4, "large"),
    ]

    sims: list[SimulationConfig] = []
    for num_resources, num_converters, room_size in combos:
        sims.append(
            SimulationConfig(
                name=f"in_context_learning_unordered/r{num_resources}_c{num_converters}_{room_size}",
                env=icl_unordered_chain_eval_env(
                    make_icl_unordered_chain_eval_env(
                        num_resources=num_resources,
                        num_converters=num_converters,
                        room_size=room_size,
                    )
                ),
            )
        )
    return sims
