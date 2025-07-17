#!/bin/bash

# Arena training with performance threshold tracking
# This recipe runs arena training and tracks when heart.gained reaches 2.0 and 5.0

echo "Starting arena training with performance threshold tracking..."

./devops/skypilot/launch.py train \
--gpus=1 \
--nodes=1 \
--no-spot \
run=$USER.recipes.arena.performance_tracking.$(date +%m-%d) \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50 \
trainer.total_timesteps=10_000_000 \
wandb.project=arena_performance_tracking \
wandb.tags=arena,performance_thresholds

echo "Training started! Performance thresholds will be tracked:"
echo "  - heart_gained_2: env_agent/heart.gained >= 2.0"
echo "  - heart_gained_5: env_agent/heart.gained >= 5.0"
echo ""
echo "Check WandB for performance_threshold/* metrics"
