#!/bin/bash

#Navigation Recipe
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

seed=$RANDOM

./devops/skypilot/launch.py train \
  run=$USER.navigation.8way.seed$seed.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/navigation/learning_progress \
  --gpus=1 \
  +trainer.env_overrides.game.num_agents=4 \
  +trainer.env_overrides.actions.move.enabled=false \
  +trainer.env_overrides.actions.rotate.enabled=false \
  +trainer.env_overrides.actions.move_8way.enabled=true \
  sim=navigation \
  seed=$seed \
  "$@"
