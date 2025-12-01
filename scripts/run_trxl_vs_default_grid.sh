#!/usr/bin/env bash
set -euo pipefail

cd /Users/sovitnayak/Documents/GitHub/metta

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RECIPES=(
  "recipes.experiment.assembly_lines.train:icl"
  "recipes.experiment.navigation.train:nav"
  "recipes.experiment.cvc.mission_variant_curriculum.train:cvc"
)

# Compare the original baseline architecture vs TRXL
ARCHS=("default" "trxl")

# Five seeds per (recipe, architecture)
SEEDS=(0 1 2 3 4)

for entry in "${RECIPES[@]}"; do
  IFS=":" read -r MODULE SHORT <<< "${entry}"

  for ARCH in "${ARCHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      TS="$(date +%Y-%m-%d_%H%M%S)"
      RUN_NAME="${SHORT}_${ARCH}_seed${SEED}_${TS}"

      echo "Launching: module=${MODULE} arch_type=${ARCH} seed=${SEED} run=${RUN_NAME}"

      uv run devops/skypilot/launch.py \
        "${MODULE}" \
        "run=${RUN_NAME}" \
        "arch_type=${ARCH}" \
        "trainer.seed=${SEED}" \
        --gpus=4 \
        --heartbeat-timeout-seconds=3600 \
        --skip-git-check

      # Small delay so Skypilot / WandB don't get spammed too hard
      sleep 1
    done
  done
done

echo "Submitted TRXL vs baseline grid for all recipes."


