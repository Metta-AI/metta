#!/bin/bash

# ColorTree Ternary (3-color) Random Curriculum BPTT Sweep
# - Uses /env/mettagrid/curriculum/colortree_easy_random
# - Sweeps bptt_horizon values and logs them in run names
# - Accepts optional seed (defaults random) and extra overrides

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

# Compose a base run label to group this batch
STAMP=$(date +%Y%m%d_%H%M%S)
BASE="${USER:-user}.colortree_ternary_random.bptt.${STAMP}"

for B in "${BPTTS[@]}"; do
  RUN_NAME="${BASE}.bptt${B}.seed${SEED}"
  echo "Launching: $RUN_NAME"
  ./devops/skypilot/launch.py train \
    run="$RUN_NAME" \
    seed="$SEED" \
    trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_random \
    +trainer.curriculum.sequence_length=2 \
    +trainer.curriculum.num_colors=3 \
    sim=colortree \
    trainer.bptt_horizon="$B" \
    "+trainer.env_overrides.game.actions.color_tree.color_to_item={0: ore_red, 1: ore_green, 2: ore_blue}" \
    "$@"
done

echo "All jobs launched."


