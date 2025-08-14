#!/bin/bash

# Navigation Task-Scaled Performance Recipe
# This recipe tests the new task-scaled performance feature using the toggle

./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  run=$USER.nav_task_scaled_perf.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.env_overrides.enable_task_perf_target=true \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  trainer.checkpoint.wandb_checkpoint_interval=50 \
  wandb.project=metta \
  wandb.group=arena_task_scaled_perf \
  wandb.name=nav_task_scaled_performance \
  "$@"
