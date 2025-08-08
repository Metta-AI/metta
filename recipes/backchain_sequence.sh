./devops/skypilot/launch.py train \
  run=$USER.backchain_sequence.bucketed.6steps.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/backchain_sequence \
  sim=backchain_sequence \
  "$@"
