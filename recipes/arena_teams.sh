./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  run=$USER.recipes.arena.teams.no.rs.$(date +%m-%d) \
  +user=krishnakanth \
  trainer.curriculum=/env/mettagrid/curriculum/arena/teams \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  "$@"

#trainer.curriculum=/env/mettagrid/curriculum/arena/teams \
