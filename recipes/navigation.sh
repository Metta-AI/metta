#!/bin/bash

#Navigation Recipe
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

./devops/skypilot/launch.py train \
run=$USER.navigation.low_reward.baseline.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/learning_progress \
--gpus=1 \
+trainer.env_overrides.game.num_agents=4 \
sim=navigation \
"$@"
