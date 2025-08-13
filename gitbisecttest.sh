set -e

# Create configs/user/gregorypylypovych.yaml with the specified contents
mkdir -p configs/user
cat > configs/user/gregorypylypovych.yaml <<EOF
# @package __global__

# This file is private property of Greg.
# DO NOT MODIFY.

run: gregorypylypovych_2025_08_12_train_process_2
# seed: 0
# sim: navigation
trainer:
  simulation:
    skip_git_check: true
    # evaluate_interval: 5
    # evaluate_remote: false
  curriculum: env/mettagrid/curriculum/autocurricula/random
  # ppo:
  #   gamma: 0.999
  # minibatch_size: 16384
  # checkpoint:
  #   checkpoint_interval: 5
  #   wandb_checkpoint_interval: 5

replay_job:
  sim:
    env: env/mettagrid/autocurricula/terrain_from_numpy
    env_overrides:
      game:
        no_agent_interference: true
EOF

# Sync dependencies
uv sync

# Activate venv if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Run the training script with the specified run argument
python3 tools/train.py +user=gregorypylypovych



