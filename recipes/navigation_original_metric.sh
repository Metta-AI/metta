#!/bin/bash

# Navigation Original Metric Recipe
# This recipe runs the navigation bucketed curriculum with the original metric for comparison
# No task-scaled performance analysis is performed

./devops/skypilot/launch.py train \
  --gpus=1 \
  --nodes=1 \
  --no-spot \
  run=$USER.nav_original_metric.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  +trainer.env_overrides.num_agents=4 \
  +trainer.env_overrides.enable_task_perf_target=false \
  sim=navigation \
  wandb.project=metta \
  wandb.group=nav_task_scaled_perf \
  wandb.name=nav_original_metric \
  "$@"
