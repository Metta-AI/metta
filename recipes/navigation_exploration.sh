#!/bin/bash

#Navigation Recipe with Exploration Tracking
#This recipe enables exploration tracking during evaluation to measure agent exploration rates
#Compare with baseline: ./recipes/navigation.sh

echo "Running navigation training with exploration tracking enabled for evaluation"

./devops/skypilot/launch.py train \
run=$USER.navigation.low_reward.exploration.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
+trainer.env_overrides.game.num_agents=4 \
+trainer.env_overrides.game.track_exploration=true \
sim=navigation \
"$@"
