

#SMOKE TEST - SINGLE TRIAL
#agents should always get 50% reward
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singletrial.smoke_test.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  --gpus=1 \
  +trainer.env_overrides.num_trials=1 \
  "$@"

#2 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.2converters.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  --gpus=1 \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#3 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.3converters.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/3_converters \
  --gpus=1 \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.4converters.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/4_converters \
  --gpus=1 \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#all
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.all.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/all \
  --gpus=1 \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"
