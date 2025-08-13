

#2 all chain lengths, no sinks
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_no_sinks.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_no_sinks \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_no_sinks.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_no_sinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

#
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.01_sinks_chainlength4.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/01_sinks_chainlength4 \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.01_sinks_chainlength4.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/01_sinks_chainlength4 \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

# 4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.4gpus.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  --gpus=4 \
  --nodes=4 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"


./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_2.128.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chain_length_2/all \
  trainer.bptt_horizon=128 \
  trainer.batch_size=1032192 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chain_length_2.256.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/chain_length_2/all \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"
