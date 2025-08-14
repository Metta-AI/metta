./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  run=$USER.contrastive_baseline_arena_recipe_skypilot.$(date +%m-%d) \
  trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  trainer.contrastive.enabled=false\
  "$@"
