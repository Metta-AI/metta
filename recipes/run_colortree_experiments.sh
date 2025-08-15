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
        # Set color_to_item mapping based on number of colors
        if [ "$num_colors" -eq 2 ]; then
            color_map="{0: ore_red, 1: ore_green}"
        else
            color_map="{0: ore_red, 1: ore_green, 2: ore_blue}"
        fi

        # Launch the run with proper env_overrides
        run_name="${USER}.colortree_${steps}step_${num_colors}colors_seed${SEED}.${STAMP}"

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
            "+trainer.env_overrides.game.actions.color_tree.color_to_item=${color_map}" \
            "+trainer.env_overrides.game.actions.color_tree.reward_mode=precise" \
            "$@"

        echo "Launched ${steps}-step with ${num_colors} colors"
        echo "---"
    done
done

echo "All runs launched!"
