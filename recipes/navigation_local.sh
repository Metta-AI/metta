#!/bin/bash

#Local Navigation Recipe (without environment context)
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

python ./tools/train.py \
--config-name user/bullm \
agent=fast \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
trainer.env_overrides.game.num_agents=4 \
sim=navigation \
agent.components.env_context.enabled=false \
run=$USER.navigation.low_reward.baseline.local.$(date +%m-%d) \
"$@"
