./tools/train.py \
  run=$USER.mazes_for_the_love_of_god_02 \
  ++trainer.env=/env/mettagrid/navigation/evals/perfect_maze_11 \
  ++trainer.simulation.evaluate_interval=10 \
  sim=perfect_mazes \
  ++replay_job.sim.env=/env/mettagrid/navigation/evals/perfect_maze_11

# USER=greg
# agent_cluster_type=positions_in_same_area
# num_rooms=4
# num_agents_per_room=4
# for seed in 0; do
#    python3 tools/train.py \
#     run=$USER.navigation.ffa_BOXES_REAL_TESTS_06 \
#     trainer.curriculum=env/mettagrid/curriculum/autocurricula/random \
#     ++trainer.ppo.gamma=0.999 \
#     ++trainer.minibatch_size=16384 \
#     ++trainer.env_overrides.special_reward_mode=ffa \
#     ++trainer.checkpoint.checkpoint_interval=5 \
#     ++trainer.checkpoint.wandb_checkpoint_interval=5 \
#     ++trainer.simulation.evaluate_interval=5 \
#     ++trainer.simulation.evaluate_remote=false \
#     seed=${seed} \
#     sim=navigation \
#     "$@"
# done
