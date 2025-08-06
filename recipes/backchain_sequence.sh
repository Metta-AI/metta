./devops/skypilot/launch.py train \
  run=$USER.backchain_sequence.baseline.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/backchain_sequence \
  --gpus=1 \
  sim=backchain_sequence \
  "$@"
