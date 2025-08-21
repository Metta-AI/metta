#!/usr/bin/env python3
"""Arena Learning Progress Test - Integrated Recipe Functionality.

This module provides a comprehensive test of the learning progress curriculum
integration with the arena environment, combining the functionality from the
original recipe into the experiments framework.
"""

import argparse
import logging
import numpy as np
import os
from typing import Dict, Any

import wandb

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressHypers
from metta.cogworks.curriculum.curriculum import Curriculum

# Import arena experiments
from experiments.arena import make_curriculum, train

# Import navigation experiments
from experiments.navigation import (
    make_curriculum as make_nav_curriculum,
    train as train_nav,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_arena_learning_progress_config() -> Dict[str, Any]:
    """Create arena configuration with learning progress curriculum.

    This function creates a comprehensive arena configuration that demonstrates
    the learning progress curriculum integration with various task buckets.
    """

    # Create arena environment
    arena = eb.make_arena(num_agents=24)

    # Disable swap action for simplicity
    arena.game.actions.swap.enabled = False

    # Create task generator for arena
    arena_tasks = cc.bucketed(arena)

    # Add various task buckets for curriculum learning
    # Agent count variations
    arena_tasks.add_bucket("game.map_builder.num_agents", [1, 2, 3, 4, 6, 12, 24])

    # Map size variations
    arena_tasks.add_bucket("game.map_builder.width", [10, 15, 20, 25, 30])
    arena_tasks.add_bucket("game.map_builder.height", [10, 15, 20, 25, 30])

    # Reward variations for different items
    for item in arena.game.inventory_item_names:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, vr.vr(0, 1.0)]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2, 3])

    # Attack cost variations
    arena_tasks.add_bucket(
        "game.actions.attack.consumed_resources.laser", [1, 10, 50, 100]
    )

    # Create learning progress algorithm hyperparameters
    lp_hypers = LearningProgressHypers(
        ema_timescale=0.001,
        progress_smoothing=0.05,
        num_active_tasks=8,
        rand_task_rate=0.25,
        sample_threshold=10,
        memory=25,
    )

    # Create curriculum with integrated learning progress algorithm
    curriculum_cfg = cc.CurriculumConfig(
        task_generator=arena_tasks,
        num_active_tasks=8,
        algorithm_hypers=lp_hypers,
    )

    return {
        "env_config": arena,
        "task_generator": arena_tasks,
        "curriculum_config": curriculum_cfg,
        "learning_progress_hypers": lp_hypers,
    }


def simulate_training_episode(
    curriculum: Curriculum, episode_num: int
) -> Dict[str, Any]:
    """Simulate a training episode with the curriculum.

    Args:
        curriculum: The curriculum instance to use for task selection
        episode_num: The episode number for logging

    Returns:
        Dictionary containing episode data
    """

    # Get a task from the curriculum
    task = curriculum.get_task()

    # Simulate task completion with random success rate
    success_rate = np.random.random()

    # Update curriculum with task performance
    curriculum.update_task_performance(task._task_id, success_rate)

    # Complete the task
    task.complete(success_rate)

    return {
        "task_id": task._task_id,
        "success_rate": success_rate,
        "episode": episode_num,
    }


def test_learning_progress_integration():
    """Test the learning progress curriculum integration."""

    print("🧪 Testing Learning Progress Curriculum Integration")

    # Test 1: Configuration creation
    print("  📋 Testing configuration creation...")
    config = create_arena_learning_progress_config()

    assert config["curriculum_config"] is not None, (
        "Curriculum config should be created"
    )
    assert config["learning_progress_hypers"] is not None, (
        "Learning progress hypers should be created"
    )

    print("    ✅ Configuration creation passed")

    # Test 2: Curriculum instantiation
    print("  🎯 Testing curriculum instantiation...")
    curriculum = config["curriculum_config"].make()

    assert curriculum is not None, "Curriculum should be instantiated"
    assert curriculum._algorithm is not None, (
        "Curriculum should have learning progress algorithm"
    )

    print("    ✅ Curriculum instantiation passed")

    # Test 3: Task generation and completion
    print("  🔄 Testing task generation and completion...")
    for i in range(10):
        episode_data = simulate_training_episode(curriculum, i)
        if i % 3 == 0:
            print(
                f"    Episode {i}: Task {episode_data['task_id']}, Success: {episode_data['success_rate']:.3f}"
            )

    print("    ✅ Task generation and completion passed")

    # Test 4: Statistics collection
    print("  📊 Testing statistics collection...")
    stats = curriculum.stats()

    assert "num_active_tasks" in stats, "Stats should include num_active_tasks"
    assert "algorithm/lp/num_active_tasks" in stats, (
        "Stats should include learning progress algorithm stats"
    )

    print(f"    ✅ Statistics collection passed: {stats}")


def test_arena_experiments_integration():
    """Test the integration with arena experiments."""

    print("🧪 Testing Arena Experiments Integration")

    # Test 1: make_curriculum with learning progress
    print("  📋 Testing make_curriculum with learning progress...")
    curriculum_cfg_lp = make_curriculum(use_learning_progress=True)

    assert curriculum_cfg_lp.algorithm_hypers is not None, (
        "Arena curriculum should use learning progress"
    )

    print("    ✅ make_curriculum with learning progress passed")

    # Test 2: make_curriculum without learning progress
    print("  📋 Testing make_curriculum without learning progress...")
    curriculum_cfg_random = make_curriculum(use_learning_progress=False)

    assert curriculum_cfg_random.algorithm_hypers is None, (
        "Arena curriculum should not use learning progress"
    )

    print("    ✅ make_curriculum without learning progress passed")

    # Test 3: train function with learning progress
    print("  🚀 Testing train function with learning progress...")
    trainer_lp = train("test_arena_lp_run", use_learning_progress=True)

    assert trainer_lp.trainer.curriculum.algorithm_hypers is not None, (
        "Arena trainer should use learning progress"
    )

    print("    ✅ train function with learning progress passed")

    # Test 4: train function without learning progress
    print("  🚀 Testing train function without learning progress...")
    trainer_random = train("test_arena_random_run", use_learning_progress=False)

    assert trainer_random.trainer.curriculum.algorithm_hypers is None, (
        "Arena trainer should not use learning progress"
    )

    print("    ✅ train function without learning progress passed")

    print("🎉 All arena experiments integration tests passed!")


def test_navigation_experiments_integration():
    """Test the integration with navigation experiments."""

    print("🧪 Testing Navigation Experiments Integration")

    # Test 1: navigation make_curriculum with learning progress
    print("  📋 Testing navigation make_curriculum with learning progress...")
    nav_curriculum_cfg_lp = make_nav_curriculum(use_learning_progress=True)

    assert nav_curriculum_cfg_lp.algorithm_hypers is not None, (
        "Navigation curriculum should use learning progress"
    )

    print("    ✅ navigation make_curriculum with learning progress passed")

    # Test 2: navigation make_curriculum without learning progress
    print("  📋 Testing navigation make_curriculum without learning progress...")
    nav_curriculum_cfg_random = make_nav_curriculum(use_learning_progress=False)

    assert nav_curriculum_cfg_random.algorithm_hypers is None, (
        "Navigation curriculum should not use learning progress"
    )

    print("    ✅ navigation make_curriculum without learning progress passed")

    # Test 3: navigation train function with learning progress
    print("  🚀 Testing navigation train function with learning progress...")
    nav_trainer_lp = train_nav("test_nav_lp_run", use_learning_progress=True)

    assert nav_trainer_lp.trainer.curriculum.algorithm_hypers is not None, (
        "Navigation trainer should use learning progress"
    )

    print("    ✅ navigation train function with learning progress passed")

    # Test 4: train function without learning progress
    print("  🚀 Testing navigation train function without learning progress...")
    nav_trainer_random = train_nav("test_nav_random_run", use_learning_progress=False)

    assert nav_trainer_random.trainer.curriculum.algorithm_hypers is None, (
        "Navigation trainer should not use learning progress"
    )

    print("    ✅ navigation train function without learning progress passed")

    print("🎉 All navigation experiments integration tests passed!")


def run_simulation_training(args):
    """Run a simulation training session with learning progress."""

    print("🚀 Running Simulation Training with Learning Progress")

    # Create configuration
    config = create_arena_learning_progress_config()

    # Create curriculum
    curriculum = config["curriculum_config"].make()

    # Setup distributed training environment variables
    if args.distributed:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(
            f"🔗 Distributed training: rank {rank}/{world_size}, local_rank {local_rank}"
        )

        # Only initialize wandb on rank 0
        if rank == 0:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "curriculum_type": "learning_progress",
                    "num_tasks": curriculum._config.num_active_tasks,
                    "learning_progress_hypers": config[
                        "learning_progress_hypers"
                    ].model_dump(),
                    "distributed": True,
                    "world_size": world_size,
                    "rank": rank,
                },
            )
    else:
        # Local training
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "curriculum_type": "learning_progress",
                "num_tasks": curriculum._config.num_active_tasks,
                "learning_progress_hypers": config[
                    "learning_progress_hypers"
                ].model_dump(),
                "distributed": False,
            },
        )

    logger.info(
        f"Starting training with {curriculum._config.num_active_tasks} active tasks"
    )

    # Training loop
    num_episodes = args.num_episodes
    stats_interval = 50

    for episode in range(num_episodes):
        # Simulate training episode
        episode_data = simulate_training_episode(curriculum, episode)

        # Log episode data (only on rank 0 for distributed training)
        if not args.distributed or int(os.environ.get("RANK", 0)) == 0:
            wandb.log(
                {
                    "episode": episode,
                    "task_id": episode_data["task_id"],
                    "success_rate": episode_data["success_rate"],
                }
            )

        # Log curriculum statistics periodically
        if episode % stats_interval == 0:
            curriculum_stats = curriculum.stats()

            if not args.distributed or int(os.environ.get("RANK", 0)) == 0:
                wandb.log(
                    {
                        "curriculum/num_active_tasks": curriculum_stats.get(
                            "num_active_tasks", 0
                        ),
                        "curriculum/num_created": curriculum_stats.get(
                            "num_created", 0
                        ),
                        "curriculum/num_evicted": curriculum_stats.get(
                            "num_evicted", 0
                        ),
                        "curriculum/num_completed": curriculum_stats.get(
                            "num_completed", 0
                        ),
                        "curriculum/num_scheduled": curriculum_stats.get(
                            "num_scheduled", 0
                        ),
                    }
                )

            logger.info(
                f"Episode {episode}: Task {episode_data['task_id']}, "
                f"Success: {episode_data['success_rate']:.3f}"
            )

    if not args.distributed or int(os.environ.get("RANK", 0)) == 0:
        wandb.finish()
    logger.info("Training completed!")


def main():
    """Main function to run all tests and optionally training."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Arena Learning Progress Test Suite")
    parser.add_argument(
        "--distributed", action="store_true", help="Run in distributed mode"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="metta", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="msb_lpdehyd_001", help="Wandb run name"
    )
    parser.add_argument(
        "--run_tests", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--run_training", action="store_true", help="Run simulation training"
    )

    args = parser.parse_args()

    # Check if we're in distributed mode
    if args.distributed:
        print("🔗 Running in distributed mode")
        print(f"   WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
        print(f"   RANK: {os.environ.get('RANK', 'Not set')}")
        print(f"   LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    else:
        print("🏠 Running in local mode")

    # Run tests if requested or if no specific mode is set
    if args.run_tests or (not args.run_training and not args.distributed):
        print("🎯 Arena & Navigation Learning Progress Test Suite")
        print("=" * 55)

        # Run integration tests
        test_learning_progress_integration()
        print()

        test_arena_experiments_integration()
        print()

        test_navigation_experiments_integration()
        print()

        # Ask user if they want to run simulation training (only in interactive mode)
        if not args.run_training and not args.distributed:
            response = (
                input("Do you want to run simulation training? (y/n): ").lower().strip()
            )
            if response in ["y", "yes"]:
                args.run_training = True

    # Run training if requested
    if args.run_training:
        print()
        run_simulation_training(args)

    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
