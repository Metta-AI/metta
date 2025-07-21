#no_clustering, right_next_to_each_other, positions_in_same_area
./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=0 \
++replay_job.sim.env=env/mettagrid/navigation/training/varied_terrain_dense \
++replay_job.sim.env_overrides.special_reward_mode=ffa \
++replay_job.sim.env_overrides.game.num_agents=1 \
++replay_job.sim.env_overrides.game.map_builder.room.agents=1 \
++replay_job.sim.env_overrides.game.map_builder.room.agent_cluster_type=no_clustering \
++replay_job.sim.env_overrides.game.map_builder.num_rooms=1 \

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=0 \
++replay_job.sim.env=env/mettagrid/navigation/training/varied_terrain_dense \
++replay_job.sim.env_overrides.special_reward_mode=ffa \
++replay_job.sim.env_overrides.game.num_agents=16 \
++replay_job.sim.env_overrides.game.map_builder.room.agents=4 \
++replay_job.sim.env_overrides.game.map_builder.room.agent_cluster_type=positions_in_same_area \
++replay_job.sim.env_overrides.game.map_builder.num_rooms=4 \

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=0 \
++replay_job.sim.env=env/mettagrid/navigation/training/varied_terrain_dense \
++replay_job.sim.env_overrides.special_reward_mode=ffa \
++replay_job.sim.env_overrides.game.num_agents=16 \
++replay_job.sim.env_overrides.game.map_builder.room.agents=4 \
++replay_job.sim.env_overrides.game.map_builder.room.agent_cluster_type=right_next_to_each_other \
++replay_job.sim.env_overrides.game.map_builder.num_rooms=4 \

# ./tools/play.py run=test-play \
# +hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
# seed=1 \
# ++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

# ./tools/play.py run=test-play \
# +hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
# seed=2 \
# ++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

# ./tools/play.py run=test-play \
# +hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
# seed=3 \
# ++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

# ./tools/play.py run=test-play \
# +hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
# seed=4 \
# ++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world
