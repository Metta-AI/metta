

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_no_sinks.0.1_progress_smoothing.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_no_sinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_one_sink.0.1_progress_smoothing.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_one_sink \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"

# 4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.0.1_progress_smoothing.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  sim=operant_conditioning_incontext \
  "$@"


./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chains_2_3_allsinks.0.1_progress_smoothing.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chains_2_3_allsinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  "$@"



./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_no_sinks.0.1_progress_smoothing.05_resource_loss_prob.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_no_sinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  trainer.env_overrides.game.resource_loss_prob=0.05 \
  "$@"

./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all_chains_one_sink.0.1_progress_smoothing.05_resource_loss_prob.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all_chains_one_sink \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  trainer.env_overrides.game.resource_loss_prob=0.05 \
  "$@"

# 4 converters
./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.all.0.1_progress_smoothing.05_resource_loss_prob.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/all \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  sim=operant_conditioning_incontext \
  trainer.env_overrides.game.resource_loss_prob=0.05 \
  "$@"


./devops/skypilot/launch.py train \
  run=$USER.operant_conditioning.in_context_learning.chains_2_3_allsinks.0.1_progress_smoothing.05_resource_loss_prob.$(date +%m-%d) \
  trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/in_context_learning/chains_2_3_allsinks \
  sim=operant_conditioning_incontext \
  trainer.bptt_horizon=256 \
  trainer.batch_size=2064384 \
  trainer.env_overrides.game.resource_loss_prob=0.05 \
  "$@"
