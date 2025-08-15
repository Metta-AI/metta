#!/bin/bash

# ColorTree Experiments Launcher for Skypilot
# Tests different step counts and color counts with proper env_overrides

set -euo pipefail

# Helper function to generate random seed
random_seed() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import random; print(random.randint(0, 100000))"
  else
    echo $(( ( (RANDOM << 15) | RANDOM ) % 100001 ))
  fi
}

# Generate random seed if not provided
SEED=${1:-$(random_seed)}
echo "Using seed: $SEED"

# Base timestamp for all runs
STAMP=$(date +%Y%m%d_%H%M%S)

# Run experiments
for steps in 16 32 64; do
    for num_colors in 2 3; do
        # Build color_to_item overrides based on number of colors
        # Using individual key overrides to avoid Hydra parsing issues with dictionary syntax
        if [ "$num_colors" -eq 2 ]; then
            color_overrides=(
                "+trainer.env_overrides.game.actions.color_tree.color_to_item.0=ore_red"
                "+trainer.env_overrides.game.actions.color_tree.color_to_item.1=ore_green"
            )
        else
            color_overrides=(
                "+trainer.env_overrides.game.actions.color_tree.color_to_item.0=ore_red"
                "+trainer.env_overrides.game.actions.color_tree.color_to_item.1=ore_green"
                "+trainer.env_overrides.game.actions.color_tree.color_to_item.2=ore_blue"
            )
        fi

        # Launch the run with proper env_overrides
        run_name="${USER}.colortree_random_${steps}step_${num_colors}colors_seed${SEED}.${STAMP}"

        echo "Launching: $run_name"
        echo "  Steps: ${steps}"
        echo "  Colors: ${num_colors}"
        echo "  Seed: ${SEED}"

        ./devops/skypilot/launch.py train \
            run=$run_name \
            seed=$SEED \
            trainer.curriculum=/env/mettagrid/curriculum/colortree_easy_random \
            +trainer.curriculum.num_colors=$num_colors \
            +trainer.curriculum.sequence_length=4 \
            sim=colortree \
            +trainer.env_overrides.game.max_steps=$steps \
            "${color_overrides[@]}" \
            "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise" \
            "$@"

        echo "Launched ${steps}-step with ${num_colors} colors"
        echo "---"
    done
done

echo "All runs launched!"
