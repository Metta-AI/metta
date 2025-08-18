


./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_no_sinks.breadcrumb_reward.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_no_sinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"


./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_one_sink.breadcrumb_reward.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_one_sink \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

# 4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.breadcrumb_reward.all$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  sim=operant_conditioning_incontext \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chains_2_3_allsinks.breadcrumb_reward.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chains_2_3_allsinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"
