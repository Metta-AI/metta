#!/bin/bash

# Arena Task-Scaled Performance Recipe
# This recipe runs the arena curriculum with task-scaled performance analysis enabled
# and a custom reward target range.
#
# +trainer.env_overrides.reward_target_min sets the minimum possible reward target for scaling (e.g., 1.0)
# +trainer.env_overrides.reward_target_max sets the maximum possible reward target for scaling (e.g., 25.0)
# The agent's mean reward will be divided by the sampled reward target (capped at 1.0) to compute task-scaled performance.

./devops/skypilot/launch.py train \
  --gpus=1 \
  --nodes=1 \
  --no-spot \
  run=$USER.recipes.arena_task_scaled_perf.8x4.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  +trainer.env_overrides.enable_task_perf_target=true \
  +trainer.env_overrides.reward_target_min=1.0 \
  +trainer.env_overrides.reward_target_max=25.0 \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  "$@"
