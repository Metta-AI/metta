#!/bin/bash

# ColorTree Binary (2-color, length-2) Training Recipe
# - Fixed target sequence and environment via /env/mettagrid/curriculum/colortree_easy_binary
# - Accepts an optional seed override (defaults to 42 if not provided)
# - Mirrors the style of recipes/navigation.sh

set -euo pipefail

# Usage:
#   ./recipes/colortree_binary.sh                 # uses default seed=42
#   ./recipes/colortree_binary.sh seed=123        # explicit seed
#   ./recipes/colortree_binary.sh --gpus=1        # pass-through to launcher (if you use one)
#   ./recipes/colortree_binary.sh +trainer.env_overrides.game.num_agents=64  # extra overrides

# Seed (Hydra top-level) — default is random
SEED=$(shuf -i 0-10000 -n 1)

# Compose a run name with timestamp
RUN_NAME="${USER:-user}.jacke.2color_10_binary_precise.$SEED.$(date +%Y%m%d_%H%M%S)"

# If a SkyPilot launcher is preferred, replace the line below with your launcher like navigation.sh does.
# Here we call the training entrypoint directly for simplicity.

./tools/train.py \
  run="$RUN_NAME" \
  seed="$SEED" \
  trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_binary \
  sim=colortree \
  "$@"


