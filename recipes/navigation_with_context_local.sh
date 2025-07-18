#!/bin/bash

#Local Navigation Recipe with Environment Context
#We expect this recipe to achieve >88% on the navigation evals with environment context injection
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

python ./tools/train.py \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
+trainer.env_overrides.game.num_agents=4 \
override sim: navigation \
agent.components.env_context.enabled=true \
run=$USER.navigation.low_reward.with_context.local.$(date +%m-%d) \
"$@"
