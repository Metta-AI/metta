#!/bin/bash

# Local navigation training script for use with tools/train.py

agent_cluster_type=positions_in_same_area
num_rooms=4
num_agents_per_room=4
seed=0
python3 tools/train.py \
run=$USER.navigation.TESTING_MASSIVE_${agent_cluster_type}_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/navigation/random \
+USER=greg \
++trainer.ppo.gamma=0.999 \
++trainer.minibatch_size=2048 \
++trainer.env_overrides.special_reward_mode=ffa \
++trainer.env_overrides.game.map_builder.room.agent_cluster_type=${agent_cluster_type} \
++trainer.env_overrides.game.num_agents=$((num_agents_per_room * num_rooms)) \
++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
seed=${seed} \
sim=navigation \
"$@"
# ++trainer.env_overrides.game.num_agents=16 \

#no_clustering, right_next_to_each_other, positions_in_same_area


# ++replay_job.sim.env_overrides.game.num_agents=8 \
# ++replay_job.sim.env_overrides.sparse_reward_top_heart_winners_every_N_steps=false \
# ++replay_job.sim.env_overrides.game.map_builder.room.agents=2 \
# ++replay_job.sim.env_overrides.game.num_agents=4 \
