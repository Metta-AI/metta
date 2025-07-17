#!/bin/bash

# Local navigation training script for use with tools/train.py

python3 tools/train.py \
run="$USER.navigation.frontier_heart_reward_only_more_rooms_test_ffa_SANDBOX.$(date +%m-%d)" \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
sim=navigation \
seed=0 \
+USER=greg \
"$@"
# ++trainer.env_overrides.game.num_agents=16 \


# ++replay_job.sim.env_overrides.game.num_agents=8 \
# ++replay_job.sim.env_overrides.sparse_reward_top_heart_winners_every_N_steps=false \
# ++replay_job.sim.env_overrides.game.map_builder.room.agents=2 \
# ++replay_job.sim.env_overrides.game.num_agents=4 \
