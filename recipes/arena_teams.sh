./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  run=$USER.recipes.arena_single_team.8x4.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/arena/teams \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  "$@"
