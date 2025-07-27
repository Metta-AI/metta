#!/bin/bash

# Local navigation training script for use with tools/train.py

# agent_cluster_type=positions_in_same_area
# num_rooms=4
# num_agents_per_room=4
# seed=0
# python3 tools/train.py \
# run=$USER.navigation.ffa_8x_red_minibatch_conc_reward_gamma_0.999_MASSIVE_${agent_cluster_type}_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
# trainer.curriculum=env/mettagrid/curriculum/navigation/random \
# +USER=greg \
# ++trainer.env_overrides.special_reward_mode=ffa \
# ++trainer.env_overrides.game.map_builder.room.agent_cluster_type=${agent_cluster_type} \
# ++trainer.env_overrides.game.num_agents=$((num_agents_per_room * num_rooms)) \
# ++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
# ++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
# seed=${seed} \
# sim=navigation \
# "$@"


agent_cluster_type=positions_in_same_area
num_rooms=4
num_agents_per_room=4
for seed in 0; do
  python3 tools/train.py \
    run=$USER.navigation.ffa_DEFAULT_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
    trainer.curriculum=env/mettagrid/curriculum/navigation/random \
    +USER=greg \
    ++trainer.ppo.gamma=0.999 \
    ++trainer.minibatch_size=16384 \
    ++trainer.env_overrides.special_reward_mode=ffa \
    ++trainer.env_overrides.game.num_agents=$((num_agents_per_room * num_rooms)) \
    ++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
    ++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
    ++trainer.env_overrides.checkpoint.checkpoint_interval=5 \
    ++trainer.env_overrides.checkpoint.wandb_checkpoint_interval=5 \
    ++trainer.env_overrides.simulation.evaluate_interval=10 \
    seed=${seed} \
    sim=navigation \
    "$@"
done
