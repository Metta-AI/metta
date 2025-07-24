#!/bin/bash

#Navigation Recipe with Environment Context
#We expect this recipe to achieve >88% on the navigation evals with environment context injection
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

./devops/skypilot/launch.py train \
run=$USER.navigation.low_reward.with_context.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
+trainer.env_overrides.game.num_agents=4 \
sim=navigation \
agent=fast_with_context \
++agent.environmental_context.enabled=true \
+trainer.batch_size=524288 \
+trainer.minibatch_size=16384 \
+trainer.forward_pass_minibatch_target_size=4096 \
+trainer.num_workers=4 \
"$@" 