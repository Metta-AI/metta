

#2 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_2.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_2 \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

#2 converters - 64 bptt
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_2.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_2 \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#3 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_3.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_3 \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_3.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_3 \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_4.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_4 \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_4.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_4 \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#all
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning.0
  0/all \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"
