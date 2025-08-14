#!/bin/bash

# ColorTree Ternary (3-color) Fixed-Sequence BPTT Sweep
# - Uses +trainer.env=env/mettagrid/colortree_easy with overrides for 3-color, 2-length sequences
# - Sweeps bptt_horizon values and logs seq/bptt/seed in run names
# - Default sequences: [2,0] and [1,2]

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

# Random default seed unless provided as seed=<N>
SEED=${seed:-$(random_seed)}

# BPTT candidates (edit as needed)
BPTTS=(4 8 16 32 64)

# Sequences to test (encode as labels without commas: 20 -> [2,0], 12 -> [1,2])
SEQ_LABELS=(20 12)

STAMP=$(date +%Y%m%d_%H%M%S)
BASE="${USER:-user}.colortree_ternary_fixed.${STAMP}"

label_to_json() {
  local lbl="$1"
  local a="${lbl:0:1}"
  local b="${lbl:1:1}"
  echo "[${a},${b}]"
}

for LBL in "${SEQ_LABELS[@]}"; do
  SEQ_JSON=$(label_to_json "$LBL")
  for B in "${BPTTS[@]}"; do
    RUN_NAME="${BASE}.seq${LBL}.bptt${B}.seed${SEED}"
    echo "Launching: $RUN_NAME"
    ./devops/skypilot/launch.py train \
      run="$RUN_NAME" \
      seed="$SEED" \
      +trainer.env=env/mettagrid/colortree_easy \
      sim=colortree \
      trainer.bptt_horizon="$B" \
      "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green, 2: ore_blue}" \
      "+trainer.env_overrides.game.actions.color_tree.target_sequence=${SEQ_JSON}" \
      "+trainer.env_overrides.game.actions.color_tree.trial_sequences=[${SEQ_JSON}]" \
      +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
      "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise" \
      "$@"
  done
done

echo "All jobs launched."


