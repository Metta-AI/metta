#!/bin/bash

# Simple ColorTree Experiments Launcher for Skypilot
# Runs experiments with 2 or 3 colors across different step counts

# Generate random seed if not provided
SEED=${1:-$RANDOM}
echo "Using seed: $SEED"

# Base command
BASE_CMD="python devops/skypilot/launch.py"

# Run experiments
for steps in 16 32 64; do
    for num_colors in 2 3; do
        # Determine config name based on steps
        if [ "$steps" -eq 64 ]; then
            config="colortree_easy"
        else
            config="colortree_easy_${steps}step"
        fi

        # Create a simple curriculum config on the fly
        curriculum_name="colortree_random_mattmagic_${num_colors}colors"
        curriculum_file="configs/env/mettagrid/curriculum/${curriculum_name}.yaml"

        cat > $curriculum_file <<EOF
_target_: metta.mettagrid.curriculum.colortree_random.ColorTreeRandomFromSetCurriculum
tasks:
  /env/mettagrid/${config}: 1.0
num_colors: $num_colors
EOF

        # Launch the run
        run_name="${USER}.colortree_${steps}step_${num_colors}colors_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

        echo "Launching: $run_name"

        $BASE_CMD \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum_name} \
            sim=${config} \
            seed=$SEED

        echo "Launched ${steps}-step with ${num_colors} colors"
        echo "---"
    done
done

echo "All runs launched!"
