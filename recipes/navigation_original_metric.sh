#!/bin/bash

# Navigation Original Metric Recipe
# This recipe runs the navigation bucketed curriculum with the original metric for comparison
# No task-scaled performance analysis is performed

./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  run=$USER.nav_original_metric.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=false \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  trainer.checkpoint.wandb_checkpoint_interval=50 \
  wandb.project=metta \
  wandb.group=arena_task_scaled_perf \
  wandb.name=nav_original_metric \
  "$@"
