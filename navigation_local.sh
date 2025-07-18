#!/bin/bash

# Local navigation training script for use with tools/train.py

python3 tools/train.py \
run="$USER.navigation.ffa_test_4rooms_of_2_right_next_to_each_other_seed0.$(date +%m-%d)" \
trainer.curriculum=env/mettagrid/curriculum/navigation/random \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.num_agents=8 \
++trainer.env_overrides.game.map_builder.room.agents=2 \
++trainer.env_overrides.game.map_builder.room.agent_cluster_type=right_next_to_each_other \
++trainer.env_overrides.game.map_builder.num_rooms=4 \
seed=0 \
sim=navigation \
+USER=greg \
"$@"
# ++trainer.env_overrides.game.num_agents=16 \

#no_clustering, right_next_to_each_other, positions_in_same_area


# ++replay_job.sim.env_overrides.game.num_agents=8 \
# ++replay_job.sim.env_overrides.sparse_reward_top_heart_winners_every_N_steps=false \
# ++replay_job.sim.env_overrides.game.map_builder.room.agents=2 \
# ++replay_job.sim.env_overrides.game.num_agents=4 \
