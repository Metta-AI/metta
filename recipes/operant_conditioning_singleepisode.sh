

#2 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.2converters.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_singleepisode \
  "$@"

#2 converters - 64 bptt
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.2converters.64.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/2_converters \
  sim=operant_conditioning_singleepisode \
  "$@"

#3 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.3converters.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/3_converters \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_singleepisode \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.3converters.64.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/3_converters \
  sim=operant_conditioning_singleepisode \
  "$@"

#4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.4converters.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/4_converters \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_singleepisode \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.4converters.64.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/4_converters \
  sim=operant_conditioning_singleepisode \
  "$@"

#all
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.all.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/all \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_singleepisode \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.singleepisode.all.64.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/all \
  sim=operant_conditioning_singleepisode \
  "$@"
