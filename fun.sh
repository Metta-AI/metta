./tools/play.py run=test-play \
+hardware=macbook cmd=play ++policy_uri=./train_dir/testing_for_andrey/checkpoints/model_0005.pt \
seed=0 \
++replay_job.sim.env=env/mettagrid/autocurricula/terrain_from_numpy \
# ++replay_job.sim.env_overrides.game.no_agent_interference=true \
