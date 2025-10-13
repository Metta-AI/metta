"""Adaptive Curriculum × Medium task complexity.

2-Axis Grid Position:
- Reward Shaping: Adaptive (learning progress-guided curriculum)
- Task Complexity: Medium (20×20 map, 20 agents, optional combat)

This configuration tests architecture performance with:
- Adaptive task selection based on learning progress
- Task variations around the medium baseline
- Tests generalization and adaptation on standard complexity tasks
"""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    CurriculumAlgorithmConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

from experiments.recipes.benchmark_architectures.adaptive import ARCHITECTURES


def make_mettagrid(num_agents: int = 20) -> MettaGridConfig:
    """Create medium complexity arena as baseline for adaptive curriculum."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    # Moderate baseline rewards (will be varied by curriculum)
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.2,
        "battery_red": 0.5,
        "laser": 0.3,
        "armor": 0.3,
        "blueprint": 0.2,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 2,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Combat will be toggled by curriculum
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create adaptive curriculum with task variations around medium baseline.

    The curriculum varies:
    - Map size (±3 from 20x20 baseline)
    - Reward values (varying difficulty of credit assignment)
    - Initial resources (sometimes provide hints)
    - Number of agents (±4 from 20 baseline)
    - Combat enabled/disabled
    """
    arena_env = arena_env or make_mettagrid()

    # Create bucketed task generator
    arena_tasks = cc.bucketed(arena_env)

    # Vary map size around medium baseline (20x20 ±3)
    arena_tasks.add_bucket("game.map_builder.width", [17, 20, 23])
    arena_tasks.add_bucket("game.map_builder.height", [17, 20, 23])

    # Vary number of agents (20 ±4)
    arena_tasks.add_bucket("game.agent.num_agents", [16, 20, 24])

    # Vary reward values for different items to create different credit assignment challenges
    arena_tasks.add_bucket(
        "game.agent.rewards.inventory.ore_red", [0.05, 0.1, 0.2, 0.3]
    )
    arena_tasks.add_bucket(
        "game.agent.rewards.inventory.battery_red", [0.3, 0.4, 0.5, 0.6]
    )
    arena_tasks.add_bucket("game.agent.rewards.inventory.laser", [0.1, 0.2, 0.3, 0.4])
    arena_tasks.add_bucket("game.agent.rewards.inventory.armor", [0.1, 0.2, 0.3, 0.4])

    # Vary reward caps to change optimization landscape
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.ore_red", [1, 2])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.battery_red", [2, 3])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.laser", [1, 2])

    # Toggle combat on/off (laser cost: 1 = enabled, 100 = disabled)
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Vary initial resources (0 = learn from scratch, 1 = get hints)
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            num_active_tasks=64,  # Larger pool for medium difficulty
            rand_task_rate=0.25,  # 25% random exploration
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=6,
            enable_detailed_slice_logging=False,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations.

    Evaluates on baseline, combat variant, and generalization test.
    """
    baseline_env = env or make_mettagrid()

    # Combat disabled variant
    no_combat_env = make_mettagrid()
    no_combat_env.game.actions.attack.consumed_resources["laser"] = 100

    # Larger variant for generalization testing
    harder_env = make_mettagrid(num_agents=24)
    harder_env.game.map_builder.width = 23
    harder_env.game.map_builder.height = 23

    return [
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_medium_baseline", env=baseline_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_medium_no_combat", env=no_combat_env
        ),
        SimulationConfig(
            suite="benchmark_arch",
            name="adaptive_medium_generalization",
            env=harder_env,
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train with adaptive curriculum on medium complexity."""
    if curriculum is None:
        curriculum = make_curriculum()

    architecture_config = ARCHITECTURES[arch_type]

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
        policy_architecture=architecture_config,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play tool."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            suite="benchmark_arch", env=eval_env, name="adaptive_medium"
        )
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            suite="benchmark_arch", env=eval_env, name="adaptive_medium"
        )
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on adaptive curriculum × medium complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
