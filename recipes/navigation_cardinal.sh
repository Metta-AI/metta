#!/bin/bash

#Navigation Recipe
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

seed=$RANDOM

./devops/skypilot/launch.py train \
  run=$USER.navigation.8way.seed$seed.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/navigation/learning_progress \
  sim=navigation \
  seed=$seed \
  '++trainer.env_overrides.game.actions.move.enabled=false' \
  '++trainer.env_overrides.game.actions.rotate.enabled=false' \
  '++trainer.env_overrides.game.actions.move_cardinal.enabled=true' \
  "$@"
