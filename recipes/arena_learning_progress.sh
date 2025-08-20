#!/bin/bash
# Arena training with Learning Progress Curriculum
# This recipe demonstrates how to use the integrated learning progress curriculum
# with the arena environment for adaptive task sampling.

set -e

echo "🚀 Starting Arena training with Learning Progress Curriculum"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Configuration
EXPERIMENT_NAME="arena_learning_progress_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
WANDB_PROJECT="metta-arena-learning-progress"

echo "📁 Output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Create configuration for arena with learning progress
cat > "${OUTPUT_DIR}/arena_lp_config.py" << 'EOF'
#!/usr/bin/env python3
"""Arena configuration with Learning Progress Curriculum."""

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.builder as eb
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressHypers

def create_arena_learning_progress_config():
    """Create arena configuration with learning progress curriculum."""

    # Create arena environment
    arena = eb.arena(num_agents=24)

    # Disable swap action for simplicity
    arena.game.actions.swap.enabled = False

    # Create task generator for arena
    arena_tasks = cc.tasks(arena)

    # Add various task buckets for curriculum learning
    # Agent count variations
    arena_tasks.add_bucket("game.level_map.num_agents", [1, 2, 3, 4, 6, 12, 24])

    # Map size variations
    arena_tasks.add_bucket("game.level_map.width", [10, 15, 20, 25, 30])
    arena_tasks.add_bucket("game.level_map.height", [10, 15, 20, 25, 30])

    # Reward variations for different items
    for item in arena.game.inventory_item_names:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, vr.vr(0, 1.0)])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2, 3])

    # Attack cost variations
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 10, 50, 100])

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
        task_generator_config=arena_tasks,
        num_active_tasks=8,
        algorithm_hypers=lp_hypers,
    )

    return {
        "env_config": arena,
        "task_generator": arena_tasks,
        "curriculum_config": curriculum_cfg,
        "learning_progress_hypers": lp_hypers,
    }

if __name__ == "__main__":
    config = create_arena_learning_progress_config()
    print("Arena Learning Progress Configuration:")
    print(f"Environment: {config['env_config'].model_dump_json(indent=2)}")
    print(f"Task Generator: {config['task_generator'].model_dump_json(indent=2)}")
    print(f"Curriculum: {config['curriculum_config'].model_dump_json(indent=2)}")
    print(f"Learning Progress: {config['learning_progress_hypers'].model_dump_json(indent=2)}")
EOF

echo "📝 Created arena learning progress configuration"

# Test the configuration
echo "🧪 Testing configuration..."
python "${OUTPUT_DIR}/arena_lp_config.py"

# Create training script
cat > "${OUTPUT_DIR}/train_arena_lp.py" << 'EOF'
#!/usr/bin/env python3
"""Training script for Arena with Learning Progress Curriculum."""

import json
import logging
import numpy as np
from pathlib import Path
import wandb

import metta.cogworks.curriculum as cc

# Import the configuration
from arena_lp_config import create_arena_learning_progress_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_training_episode(curriculum, episode_num):
    """Simulate a training episode with the curriculum."""

    # Get a task from the curriculum
    task = curriculum.get_task()
    task_id = task._task_id

    # Simulate task completion with varying success rates
    # In real training, this would come from the actual environment
    base_success = 0.3 + 0.4 * np.sin(episode_num * 0.1 + task_id * 0.5)
    noise = np.random.normal(0, 0.1)
    success_rate = np.clip(base_success + noise, 0.0, 1.0)

    # Complete the task
    task.complete(success_rate)

    # Update curriculum algorithm with task performance
    curriculum.update_task_performance(task_id, success_rate)

    return {
        "episode": episode_num,
        "task_id": task_id,
        "success_rate": success_rate,
        "env_config": task.get_env_cfg().model_dump(),
    }

def train_with_learning_progress():
    """Train with learning progress curriculum."""

    # Create configuration
    config = create_arena_learning_progress_config()

    # Create curriculum
    curriculum = config["curriculum_config"].make()

    # Initialize wandb
    wandb.init(
        project="metta-arena-learning-progress",
        config={
            "curriculum_type": "learning_progress",
            "num_tasks": curriculum._config.num_active_tasks,
            "learning_progress_hypers": config["learning_progress_hypers"].model_dump(),
        }
    )

    logger.info(f"Starting training with {curriculum._config.num_active_tasks} active tasks")

    # Training loop
    num_episodes = 1000
    stats_interval = 50

    for episode in range(num_episodes):

        # Simulate training episode
        episode_data = simulate_training_episode(curriculum, episode)

        # Log episode data
        wandb.log({
            "episode": episode,
            "task_id": episode_data["task_id"],
            "success_rate": episode_data["success_rate"],
        })

        # Log curriculum statistics periodically
        if episode % stats_interval == 0:
            curriculum_stats = curriculum.stats()
            wandb.log({
                "curriculum/num_active_tasks": curriculum_stats.get("num_active_tasks", 0),
                "curriculum/num_created": curriculum_stats.get("num_created", 0),
                "curriculum/num_evicted": curriculum_stats.get("num_evicted", 0),
                "curriculum/num_completed": curriculum_stats.get("num_completed", 0),
                "curriculum/num_scheduled": curriculum_stats.get("num_scheduled", 0),
            })

            logger.info(f"Episode {episode}: Task {episode_data['task_id']}, "
                       f"Success: {episode_data['success_rate']:.3f}")

    wandb.finish()
    logger.info("Training completed!")

if __name__ == "__main__":
    train_with_learning_progress()
EOF

echo "📝 Created training script"

# Make scripts executable
chmod +x "${OUTPUT_DIR}/arena_lp_config.py"
chmod +x "${OUTPUT_DIR}/train_arena_lp.py"

# Run integration test first
echo "🧪 Running integration test..."
cd metta/cogworks/curriculum
python -m pytest test_comprehensive.py test_integration.py -v
cd ../../..

echo "✅ Integration test passed!"

# Run training
echo "🚀 Starting training..."
cd "${OUTPUT_DIR}"
python train_arena_lp.py

echo "🎉 Training completed!"
echo "📊 Check wandb for training metrics: https://wandb.ai/your-username/${WANDB_PROJECT}"
echo "📁 Results saved in: ${OUTPUT_DIR}"
