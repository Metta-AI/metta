#!/bin/bash

# Clean ColorTree Experiments Launcher
# Uses pre-created curriculum files for each step/color combination

# Generate random seed if not provided
SEED=${1:-$RANDOM}
echo "Using seed: $SEED"
echo ""

# Run experiments
for steps in 16 32 64; do
    for num_colors in 2 3 4 5; do
        # Determine curriculum name based on steps
        if [ "$steps" -eq 64 ]; then
            curriculum="colortree_easy_${num_colors}colors"
        else
            curriculum="colortree_easy_${steps}step_${num_colors}colors"
        fi

        # Launch the run
        run_name="${USER}.colortree_${steps}step_${num_colors}colors_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

        echo "Launching: $run_name"
        echo "  Steps: ${steps}"
        echo "  Colors: ${num_colors}"
        echo "  Curriculum: ${curriculum}"
        echo "  Seed: ${SEED}"

        python devops/skypilot/launch.py train \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            seed=$SEED

        echo "---"
    done
done
