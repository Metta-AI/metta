#!/bin/bash

# Navigation Task-Scaled Performance with Custom Range Recipe
# This recipe demonstrates using configurable reward target ranges (1.0 to 25.0)
# instead of the default range (0.0 to 10.0)

./devops/skypilot/launch.py train \
  --gpus=1 \
  --nodes=1 \
  --no-spot \
  run=$USER.nav_task_scaled_perf_custom_range.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  +trainer.env_overrides.num_agents=4 \
  +trainer.env_overrides.enable_task_perf_target=true \
  +trainer.env_overrides.reward_target_min=1.0 \
  +trainer.env_overrides.reward_target_max=5.0 \
  sim=navigation \
  wandb.project=metta \
  wandb.group=nav_task_scaled_perf \
  wandb.name=nav_task_scaled_performance_custom_range \
  "$@"
