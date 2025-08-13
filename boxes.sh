USER=greg
agent_cluster_type=positions_in_same_area
num_rooms=4
num_agents_per_room=4
for seed in 0; do
   python3 tools/train.py \
    run=$USER.navigation.ffa_BOXES_TEST5_FOR_REAL \
    trainer.curriculum=env/mettagrid/curriculum/autocurricula/random \
    ++trainer.ppo.gamma=0.999 \
    ++trainer.env_overrides.special_reward_mode=ffa \
    ++trainer.checkpoint.checkpoint_interval=5 \
    ++trainer.checkpoint.wandb_checkpoint_interval=5 \
    ++trainer.simulation.evaluate_interval=5 \
    ++trainer.simulation.evaluate_remote=false \
    seed=${seed} \
    sim=navigation \
    "$@"
done


# USER=greg
# agent_cluster_type=positions_in_same_area
# num_rooms=4
# num_agents_per_room=4
# for seed in 0 1 2 3 4; do
#    ./devops/skypilot/launch.py train \
#     run=$USER.navigation.ffa_BOXES_500_const_penalty0.1_PASSTHROUGH_seed${seed}.$(date +%m-%d) \
#     trainer.curriculum=env/mettagrid/curriculum/autocurricula/random \
#     --gpus=1 \
#     --nodes=1 \
#     --no-spot \
#     ++trainer.ppo.gamma=0.977 \
#     ++trainer.minibatch_size=16384 \
#     ++trainer.env_overrides.special_reward_mode=ffa \
#     ++trainer.checkpoint.checkpoint_interval=200 \
#     ++trainer.checkpoint.wandb_checkpoint_interval=200 \
#     ++trainer.simulation.evaluate_interval=200 \
#     seed=${seed} \
#     sim=navigation \
#     "$@"
# done
