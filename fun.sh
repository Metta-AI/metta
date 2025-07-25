./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=0 \
++replay_job.sim.env_overrides.game.num_agents=8 \
++replay_job.sim.env_overrides.game.map_builder.room.agents=2 \
++replay_job.sim.env_overrides.game.map_builder.num_rooms=4 \
++replay_job.sim.env=env/mettagrid/autocurriucla/defaults \
++replay_job.sim.env_overrides.game.max_steps=1000

# ./tools/play.py run=test-play \
# +hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
# seed=0 \
# ++replay_job.sim.env_overrides.game.num_agents=8 \
# ++replay_job.sim.env_overrides.game.map_builder.room.agents=2 \
# ++replay_job.sim.env_overrides.game.map_builder.num_rooms=4 \
# ++replay_job.sim.env=env/mettagrid/navigation/training/defaults \
# ++replay_job.sim.env_overrides.game.max_steps=1000

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
