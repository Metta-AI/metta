#!/bin/bash

#Navigation Recipe
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

# Set default value for movement metrics
MOVEMENT_METRICS=${MOVEMENT_METRICS:-false}

./devops/skypilot/launch.py train \
run=$USER.navigation.low_reward.baseline.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
+trainer.env_overrides.game.num_agents=4 \
+trainer.env_overrides.game.track_movement_metrics=$MOVEMENT_METRICS \
sim=navigation \
"$@"
