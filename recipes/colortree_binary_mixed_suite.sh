#!/bin/bash

# ColorTree Binary (2-color) Mixed Suite
# - Mix fixed sequences and random curriculum jobs in one shot
# - Sequences: [1,0], [0,1]; Random: colortree_easy_random with num_colors=2, length=2
# - BPTT set you can tune; embeds seq/random, bptt, seed in run names

set -euo pipefail

random_seed() {
  if command -v jot >/dev/null 2>&1; then
    jot -r 1 0 100000
  elif command -v gshuf >/dev/null 2>&1; then
    gshuf -i 0-100000 -n 1
  elif command -v shuf >/dev/null 2>&1; then
    shuf -i 0-100000 -n 1
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import random
print(random.randint(0, 100000))
PY
  else
    echo $(( ( (RANDOM << 15) | RANDOM ) % 100001 ))
  fi
}

SEED=${seed:-$(random_seed)}
BPTTS=(8 16 32 64)

STAMP=$(date +%Y%m%d_%H%M%S)
BASE="${USER:-user}.colortree_binary_mixed.${STAMP}"

# Fixed sequences
FIXED_LABELS=(10 01)

lbl_to_json() {
  local lbl="$1"
  local a="${lbl:0:1}"
  local b="${lbl:1:1}"
  echo "[${a},${b}]"
}

for B in "${BPTTS[@]}"; do
  # Random binary job
  RUN_NAME_RAND="${BASE}.random.bptt${B}.seed${SEED}"
  echo "Launching: $RUN_NAME_RAND"
  ./devops/skypilot/launch.py train \
    run="$RUN_NAME_RAND" \
    seed="$SEED" \
    trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_random \
    +trainer.curriculum.sequence_length=2 \
    +trainer.curriculum.num_colors=2 \
    sim=colortree \
    trainer.bptt_horizon="$B" \
    "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green}" \
    "$@"

  # Fixed binary jobs
  for LBL in "${FIXED_LABELS[@]}"; do
    SEQ_JSON=$(lbl_to_json "$LBL")
    RUN_NAME_FIXED="${BASE}.seq${LBL}.bptt${B}.seed${SEED}"
    echo "Launching: $RUN_NAME_FIXED"
    ./devops/skypilot/launch.py train \
      run="$RUN_NAME_FIXED" \
      seed="$SEED" \
      +trainer.env=env/mettagrid/colortree_easy \
      sim=colortree \
      trainer.bptt_horizon="$B" \
      "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green}" \
      "+trainer.env_overrides.game.actions.color_tree.target_sequence=${SEQ_JSON}" \
      "+trainer.env_overrides.game.actions.color_tree.trial_sequences=[${SEQ_JSON}]" \
      +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
      "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise" \
      "$@"
  done
done

echo "All jobs launched."


