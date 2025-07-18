#!/bin/bash

#no_clustering, right_next_to_each_other, positions_in_same_area

agent_cluster_type=positions_in_same_area
num_rooms=4
num_agents_per_room=2
for seed in 0; do
  ./devops/skypilot/launch.py train \
    run=$USER.navigation.ffa_MASSIVE_${agent_cluster_type}_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
    trainer.curriculum=env/mettagrid/curriculum/navigation/random \
    --gpus=4 \
    --nodes=8 \
    --no-spot \
    +USER=greg \
    ++trainer.env_overrides.special_reward_mode=ffa \
    ++trainer.env_overrides.game.map_builder.room.agent_cluster_type=${agent_cluster_type} \
    ++trainer.env_overrides.game.num_agents=$((num_agents_per_room * num_rooms)) \
    ++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
    ++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
    seed=${seed} \
    sim=navigation \
    "$@"
done
