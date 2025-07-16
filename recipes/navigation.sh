#!/bin/bash

#Navigation Recipe
#We expect this recipe to achieve >88% on the navigation evals
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25

# ./devops/skypilot/launch.py train \
# run=$USER.navigation.low_reward.baseline.$(date +%m-%d) \
# trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
# --gpus=1 \
# +USER=greg \
# +trainer.env_overrides.game.num_agents=4 \
# sim=navigation \
# "$@"

./devops/skypilot/launch.py train \
run=$USER.navigation.frontier_heart_reward_only_2_per_room.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
+USER=greg \
++trainer.env_overrides.sparse_reward_top_heart_winners_every_N_steps=true \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
sim=navigation \
"$@"
