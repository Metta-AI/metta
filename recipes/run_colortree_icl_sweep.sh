#!/bin/bash

# ColorTree ICL Capability Sweep
# Tests the limits of In-Context Learning with various configurations
# Sweeps: steps (128, 64), BPTT (256, 128, 64), colors (2, 3, 4)

# Generate random seed if not provided
SEED=${1:-$RANDOM}
USE_USER_CONFIG=${2:-"no"}  # Pass "user" as second arg to include user config

echo "=== ColorTree ICL Capability Sweep ==="
echo "Testing: 2 step sizes × 3 BPTT horizons × 3 color counts = 18 experiments"
echo "Using seed: $SEED"
if [ "$USE_USER_CONFIG" = "user" ]; then
    echo "Including user config: user/jacke"
fi
echo ""

# Build command with BPTT-dependent batch size
build_command() {
    local run_name=$1
    local curriculum=$2
    local bptt=$3

    # Calculate appropriate batch size based on BPTT horizon
    local num_agents=8192
    local batch_size=524288  # Default for BPTT=64
    if [ $bptt -eq 128 ]; then
        batch_size=1048576  # 1M
    elif [ $bptt -eq 256 ]; then
        batch_size=2097152  # 2M
    fi

    if [ "$USE_USER_CONFIG" = "user" ]; then
        echo "python devops/skypilot/launch.py train user=jacke \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            trainer.batch_size=${batch_size} \
            seed=$SEED"
    else
        echo "python devops/skypilot/launch.py train \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            trainer.batch_size=${batch_size} \
            seed=$SEED"
    fi
}

# Counter for experiment numbering
exp_num=1

# Main sweep loops
for steps in 128 64; do
    for bptt in 256 128 64; do
        for num_colors in 2 3 4; do
            # Determine curriculum name based on steps
            # For 64 steps, curriculum files don't include step count
            if [ "$steps" -eq 64 ]; then
                curriculum="colortree_easy_${num_colors}colors"
            else
                # 128-step curricula now exist for 2, 3, and 4 colors
                curriculum="colortree_easy_${steps}step_${num_colors}colors"
            fi

            # Calculate BPTT/steps ratio for display
            ratio=$(python -c "print(f'{$bptt/$steps:.1f}')")

            # Build run name with all parameters
            run_name="${USER}.icl_${steps}step_${num_colors}col_bptt${bptt}_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

            echo "Experiment ${exp_num}/18: ${steps} steps, ${num_colors} colors, BPTT=${bptt}"
            echo "  Run name: $run_name"
            echo "  BPTT/Steps ratio: ${ratio}x"
            echo "  Curriculum: ${curriculum}"

            eval $(build_command "$run_name" "$curriculum" "$bptt")

            echo "---"
            exp_num=$((exp_num + 1))
        done
    done
done

echo ""
echo "=== ICL Sweep Complete ==="
echo "BPTT/Steps ratios tested:"
echo "  0.5x: BPTT shorter than episode"
echo "  1.0x: BPTT equals episode length"
echo "  2.0x: BPTT spans 2 episodes (previously found optimal)"
echo "  4.0x: BPTT spans 4 episodes"
echo ""
echo "All experiments use standard learning rate (no scaling)"
