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
agent_cluster_type=right_next_to_each_other
num_rooms=4
num_agents_per_room=2
for seed in 0 1 2 3 4; do
  ./devops/skypilot/launch.py train \
    run=$USER.navigation.ffa_${agent_cluster_type}_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
    trainer.curriculum=env/mettagrid/curriculum/navigation/random \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    +USER=greg \
    ++trainer.env_overrides.special_reward_mode=ffa \
    ++trainer.env_overrides.game.map_builder.room.agent_cluster_type=${agent_cluster_type} \
    ++trainer.env_overrides.game.num_agents=${$(num_agents_per_room * num_rooms)} \
    ++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
    ++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
    seed=${seed} \
    sim=navigation \
    "$@"
done
