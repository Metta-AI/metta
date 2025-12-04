"""
Example recipes demonstrating regret-based curriculum learning.

Based on ACCEL paper: "Adversarially Compounding Complexity by Editing Levels"
Reference: https://accelagent.github.io/

This module provides two main regret-based approaches:
1. PrioritizedRegret: Prioritize tasks with highest regret (furthest from optimal)
2. RegretLearningProgress: Prioritize tasks where regret is decreasing fastest
"""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
)
from metta.cogworks.curriculum.prioritized_regret_algorithm import PrioritizedRegretConfig
from metta.cogworks.curriculum.regret_learning_progress_algorithm import RegretLearningProgressConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def make_arena_env(num_agents: int = 24) -> MettaGridConfig:
    """Create an arena environment for testing regret-based curricula."""
    arena_env = eb.make_arena(num_agents=num_agents)

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
    }

    return arena_env


def make_prioritized_regret_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    optimal_value: float = 1.0,
    temperature: float = 1.0,
) -> CurriculumConfig:
    """
    Create curriculum using PrioritizedRegret algorithm.

    Strategy: "Go to tasks with highest regret"
    - Prioritizes tasks where agent is furthest from optimal performance
    - Maintains tasks at the frontier of agent capabilities
    - Good for maintaining challenge and preventing forgetting

    Args:
        arena_env: Environment configuration
        enable_detailed_slice_logging: Enable detailed statistics logging
        optimal_value: Maximum achievable score (typically 1.0)
        temperature: Softmax temperature for task sampling (higher = more random)

    Returns:
        Curriculum configuration with PrioritizedRegret algorithm
    """
    arena_env = arena_env or make_arena_env()

    arena_tasks = cc.bucketed(arena_env)

    # Add task variation buckets
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Toggle combat difficulty
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    algorithm_config = PrioritizedRegretConfig(
        optimal_value=optimal_value,
        regret_ema_timescale=0.01,  # Decay rate for regret EMA
        exploration_bonus=0.1,  # Bonus for unexplored tasks
        temperature=temperature,  # Sampling temperature
        min_samples_for_prioritization=2,  # Min samples before using regret
        max_memory_tasks=1000,
        max_slice_axes=5,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_regret_learning_progress_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    optimal_value: float = 1.0,
    invert_regret_progress: bool = True,
) -> CurriculumConfig:
    """
    Create curriculum using RegretLearningProgress algorithm.

    Strategy: "Go to tasks where regret is getting lower fastest"
    - Prioritizes tasks where agent is learning/improving most rapidly
    - Identifies tasks at the "learning frontier"
    - Good for accelerating learning on productive tasks

    Args:
        arena_env: Environment configuration
        enable_detailed_slice_logging: Enable detailed statistics logging
        optimal_value: Maximum achievable score (typically 1.0)
        invert_regret_progress: Prioritize decreasing regret (True) vs any change (False)

    Returns:
        Curriculum configuration with RegretLearningProgress algorithm
    """
    arena_env = arena_env or make_arena_env()

    arena_tasks = cc.bucketed(arena_env)

    # Add task variation buckets
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Toggle combat difficulty
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    algorithm_config = RegretLearningProgressConfig(
        optimal_value=optimal_value,
        regret_ema_timescale=0.001,  # Decay rate for regret EMA
        use_bidirectional=True,  # Use fast/slow EMA comparison
        exploration_bonus=0.1,
        progress_smoothing=0.05,
        invert_regret_progress=invert_regret_progress,  # Prioritize decreasing regret
        min_samples_for_lp=2,
        max_memory_tasks=1000,
        max_slice_axes=5,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations."""
    basic_env = env or make_arena_env()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="regret_arena", name="basic", env=basic_env),
        SimulationConfig(suite="regret_arena", name="combat", env=combat_env),
    ]


def train_prioritized_regret(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train with PrioritizedRegret curriculum.

    This trains an agent that prioritizes tasks with highest regret,
    following the ACCEL principle of maintaining tasks at the frontier
    of agent capabilities.

    Run with:
        ./tools/run.py recipes.experiment.regret_examples.train_prioritized_regret
    """
    curriculum = curriculum or make_prioritized_regret_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(losses=LossesConfig(ppo=PPOConfig()))
    policy_config = policy_architecture or ViTSlidingTransConfig()
    training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    evaluator = EvaluatorConfig(simulations=eval_simulations)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=evaluator,
        policy_architecture=policy_config,
    )


def train_regret_learning_progress(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train with RegretLearningProgress curriculum.

    This trains an agent that prioritizes tasks where regret is decreasing
    fastest, identifying tasks at the "learning frontier" where the agent
    is improving most rapidly.

    Run with:
        ./tools/run.py recipes.experiment.regret_examples.train_regret_learning_progress
    """
    curriculum = curriculum or make_regret_learning_progress_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(losses=LossesConfig(ppo=PPOConfig()))
    policy_config = policy_architecture or ViTSlidingTransConfig()
    training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    evaluator = EvaluatorConfig(simulations=eval_simulations)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=evaluator,
        policy_architecture=policy_config,
    )


def evaluate(run: str = "local.regret_test.1") -> EvaluateTool:
    """Evaluate a trained regret-based model."""
    return EvaluateTool(
        policy_uris=[f"wandb://run/{run}"],
        simulations=make_evals(),
    )


# Comparison experiment: Train with all three approaches
def compare_curricula(enable_detailed_slice_logging: bool = False) -> dict[str, TrainTool]:
    """
    Create training configs for comparing different curriculum approaches:
    1. Standard Learning Progress (baseline)
    2. Prioritized Regret (highest regret)
    3. Regret Learning Progress (fastest regret decrease)

    Returns:
        Dictionary mapping approach names to TrainTool configs
    """
    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

    arena_env = make_arena_env()

    # Create base curriculum structure
    arena_tasks = cc.bucketed(arena_env)
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # 1. Standard Learning Progress
    lp_curriculum = arena_tasks.to_curriculum(
        algorithm_config=LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )
    )

    # 2. Prioritized Regret
    pr_curriculum = arena_tasks.to_curriculum(
        algorithm_config=PrioritizedRegretConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            exploration_bonus=0.1,
            temperature=1.0,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )
    )

    # 3. Regret Learning Progress
    rlp_curriculum = arena_tasks.to_curriculum(
        algorithm_config=RegretLearningProgressConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.001,
            use_bidirectional=True,
            exploration_bonus=0.1,
            invert_regret_progress=True,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )
    )

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(losses=LossesConfig(ppo=PPOConfig()))
    policy_config = ViTSlidingTransConfig()
    evaluator = EvaluatorConfig(simulations=eval_simulations)

    return {
        "learning_progress": TrainTool(
            trainer=trainer_cfg,
            training_env=TrainingEnvironmentConfig(curriculum=lp_curriculum),
            evaluator=evaluator,
            policy_architecture=policy_config,
        ),
        "prioritized_regret": TrainTool(
            trainer=trainer_cfg,
            training_env=TrainingEnvironmentConfig(curriculum=pr_curriculum),
            evaluator=evaluator,
            policy_architecture=policy_config,
        ),
        "regret_learning_progress": TrainTool(
            trainer=trainer_cfg,
            training_env=TrainingEnvironmentConfig(curriculum=rlp_curriculum),
            evaluator=evaluator,
            policy_architecture=policy_config,
        ),
    }
