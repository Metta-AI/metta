

#2 converters - memory 128
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.forking.2_converters.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_forking \
  "$@"

#2 converters - memory 256

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.forking.2_converters.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  sim=operant_conditioning_forking \
  "$@"
