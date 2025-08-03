USER=greg
agent_cluster_type=positions_in_same_area
num_rooms=4
num_agents_per_room=4
for seed in 0; do
   python3 tools/train.py \
    run=$USER.navigation.ffa_building_to_BOXES_v5 \
    trainer.curriculum=env/mettagrid/curriculum/autocurricula/random \
    ++trainer.ppo.gamma=0.999 \
    ++trainer.minibatch_size=16384 \
    ++trainer.env_overrides.special_reward_mode=ffa \
    ++trainer.checkpoint.checkpoint_interval=5 \
    ++trainer.checkpoint.wandb_checkpoint_interval=5 \
    ++trainer.simulation.evaluate_interval=5 \
    seed=${seed} \
    sim=navigation \
    "$@"
done
# ++trainer.env_overrides.game.map_builder.root.params.rows=2 \
#     ++trainer.env_overrides.game.map_builder.root.params.columns=2 \
#     ++trainer.env_overrides.game.map_builder.root.children.0.scene.params.agents=${num_agents_per_room} \
#agent_cluster_type=positions_in_same_area
# num_rooms=4
# num_agents_per_room=4
# for seed in 0; do
#    ./devops/skypilot/launch.py train \
#     run=$USER.navigation.ffa_NAV_BOXES_maxobs_${num_rooms}rooms_of_${num_agents_per_room}_seed${seed}.$(date +%m-%d) \
#     trainer.curriculum=env/mettagrid/curriculum/autocurricula/random \
#     ++trainer.env=env/mettagrid/autocurricula/defaults \
#     --gpus=1 \
#     --nodes=1 \
#     --no-spot \
#     +USER=greg \
#     ++trainer.ppo.gamma=0.999 \
#     ++trainer.minibatch_size=16384 \
#     ++trainer.env_overrides.special_reward_mode=ffa \
#     ++trainer.env_overrides.game.num_agents=$((num_agents_per_room * num_rooms)) \
#     ++trainer.env_overrides.game.map_builder.room.agents=${num_agents_per_room} \
#     ++trainer.env_overrides.game.map_builder.num_rooms=${num_rooms} \
#     ++trainer.checkpoint.checkpoint_interval=50 \
#     ++trainer.checkpoint.wandb_checkpoint_interval=50 \
#     ++trainer.simulation.evaluate_interval=50 \
#     seed=${seed} \
#     sim=navigation \
#     "$@"
# done

