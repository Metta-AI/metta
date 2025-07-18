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
run=$USER.navigation.ffa_random_spawn_4rooms_of_2_seed0.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
--nodes=1 \
--no-spot \
+USER=greg \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=0 \
sim=navigation \
"$@"

./devops/skypilot/launch.py train \
run=$USER.navigation.ffa_random_spawn_4rooms_of_2_seed1.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
--nodes=1 \
--no-spot \
+USER=greg \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=1 \
sim=navigation \
"$@"

./devops/skypilot/launch.py train \
run=$USER.navigation.ffa_random_spawn_4rooms_of_2_seed2.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
--nodes=1 \
--no-spot \
+USER=greg \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=2 \
sim=navigation \
"$@"

./devops/skypilot/launch.py train \
run=$USER.navigation.ffa_random_spawn_4rooms_of_2_seed3.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
--nodes=1 \
--no-spot \
+USER=greg \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=3 \
sim=navigation \
"$@"

./devops/skypilot/launch.py train \
run=$USER.navigation.ffa_random_spawn_4rooms_of_2_seed4.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
--gpus=1 \
--nodes=1 \
--no-spot \
+USER=greg \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=4 \
sim=navigation \
"$@"
