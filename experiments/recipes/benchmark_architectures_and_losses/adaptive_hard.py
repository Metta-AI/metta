"""Adaptive Curriculum × Hard task complexity.

2-Axis Grid Position:
- Reward Shaping: Adaptive (learning progress-guided curriculum)
- Task Complexity: Hard (25×25 map, 24 agents, full combat)

This configuration tests architecture performance with:
- Adaptive task selection based on learning progress
- Task variations around the hard baseline
- Tests generalization and adaptation on challenging tasks
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

from experiments.recipes.benchmark_architectures.benchmark import ARCHITECTURES


def make_mettagrid(num_agents: int = 20) -> MettaGridConfig:
    """Create arena with adaptive curriculum and hard task complexity baseline.

    Baseline (Hard):
    - 3:1 converter ratio (fixed for hard)
    - No initial resources
    - Standard 20×20 map, 20 agents

    Curriculum will vary parameters around this baseline.
    """
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size (will be varied by curriculum)
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    # Sparse baseline rewards (will be varied by curriculum)
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.05,
        "battery_red": 0.2,
        "laser": 0.1,
        "armor": 0.1,
        "blueprint": 0.05,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 2,
        "battery_red": 2,
        "laser": 2,
        "armor": 2,
        "blueprint": 2,
    }

    # Hard task complexity: 3:1 converter (fixed for hard, not varied)
    # No need to modify - this is the default converter ratio

    # Combat enabled (standard)
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create adaptive curriculum with task variations around hard baseline.

    The curriculum varies:
    - Map size (±3 from 25x25 baseline)
    - Reward values (from terminal-only to sparse)
    - Initial resources (sometimes provide hints)
    - Number of agents (±4 from 24 baseline)
    - Combat enabled/disabled
    """
    arena_env = arena_env or make_mettagrid()

    # Create bucketed task generator
    arena_tasks = cc.bucketed(arena_env)

    # Vary map size around standard baseline (20x20 ±3)
    arena_tasks.add_bucket("game.map_builder.width", [17, 20, 23])
    arena_tasks.add_bucket("game.map_builder.height", [17, 20, 23])

    # Vary number of agents (20 ±4)
    arena_tasks.add_bucket("game.agent.num_agents", [16, 20, 24])

    # Vary reward values - from terminal-only to sparse to moderate
    # This creates different levels of credit assignment difficulty
    arena_tasks.add_bucket(
        "game.agent.rewards.inventory.ore_red", [0.0, 0.01, 0.05, 0.1]
    )
    arena_tasks.add_bucket(
        "game.agent.rewards.inventory.battery_red", [0.0, 0.05, 0.1, 0.2, 0.3]
    )
    arena_tasks.add_bucket("game.agent.rewards.inventory.laser", [0.0, 0.05, 0.1, 0.2])
    arena_tasks.add_bucket("game.agent.rewards.inventory.armor", [0.0, 0.05, 0.1, 0.2])

    # Vary reward caps
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.ore_red", [1, 2])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.battery_red", [2, 3, 4])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.laser", [1, 2])
    arena_tasks.add_bucket("game.agent.rewards.inventory_max.armor", [1, 2])

    # Toggle combat on/off (laser cost: 1 = enabled, 100 = disabled)
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Vary initial resources (0 = learn from scratch, 1 = get hints, 2 = more help)
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1, 2])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            num_active_tasks=128,  # Large pool for hard difficulty
            rand_task_rate=0.3,  # 30% random exploration (higher for harder tasks)
            ema_timescale=0.001,
            exploration_bonus=0.15,  # Higher exploration bonus for hard tasks
            max_memory_tasks=2000,
            max_slice_axes=7,
            enable_detailed_slice_logging=False,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations.

    Evaluates on baseline, combat variant, no combat variant, and generalization tests.
    """
    baseline_env = env or make_mettagrid()

    # Combat disabled variant
    no_combat_env = make_mettagrid()
    no_combat_env.game.actions.attack.consumed_resources["laser"] = 100

    # Terminal-only rewards (maximum difficulty)
    terminal_env = make_mettagrid()
    terminal_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.0,
        "battery_red": 0.0,
        "laser": 0.0,
        "armor": 0.0,
        "blueprint": 0.0,
    }

    # Smaller variant for generalization testing
    easier_env = make_mettagrid(num_agents=20)
    easier_env.game.map_builder.width = 22
    easier_env.game.map_builder.height = 22

    return [
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_hard_baseline", env=baseline_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_hard_no_combat", env=no_combat_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_hard_terminal", env=terminal_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="adaptive_hard_generalization", env=easier_env
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train with adaptive curriculum on hard complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="adaptive_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="adaptive_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on adaptive curriculum × hard complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
