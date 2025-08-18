./devops/skypilot/launch.py train \
  --gpus=4 \
  --nodes=8 \
  --no-spot \
  --skip-git-check \
  run=$USER.recipes.coop_rewardsharing_v2.8x4.$(date +%m-%d_%H-%M-%S) \
  trainer.curriculum=/env/mettagrid/curriculum/coop_rewardsharing/learning_progress \
  trainer.optimizer.learning_rate=0.0045 \
  trainer.optimizer.type=muon \
  trainer.simulation.evaluate_interval=50 \
  "$@"
