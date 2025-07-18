./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=0 \
++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=1 \
++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=2 \
++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=3 \
++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world

./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/gregorypylypovych.navigation.winners_rewarded_every_10_steps.07-15/checkpoints/model_0000.pt \
seed=4 \
++replay_job.sim.env=env/mettagrid/navigation/training/cylinder_world
