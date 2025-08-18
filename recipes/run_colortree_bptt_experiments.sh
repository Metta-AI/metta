#!/bin/bash

# ColorTree BPTT Horizon Experiments
# Tests different BPTT horizons for various step configurations
u# Note: Uses existing curriculum files and can optionally use user config

# Generate random seed if not provided
SEED=${1:-$RANDOM}
USE_USER_CONFIG=${2:-"no"}  # Pass "user" as second arg to include user config
echo "Using seed: $SEED"
if [ "$USE_USER_CONFIG" = "user" ]; then
    echo "Including user config: user/jacke"
fi
echo ""

# Test configurations:
# 1. 128 steps with BPTT horizons of 64, 128, and 256
# 2. 64 steps with BPTT horizon of 128
# 3. 32 steps with BPTT horizon of 64

# Fixed parameters
NUM_COLORS=2

# Build base command with optional user config
build_command() {
    local run_name=$1
    local curriculum=$2
    local bptt=$3

    if [ "$USE_USER_CONFIG" = "user" ]; then
        echo "python devops/skypilot/launch.py train user=jacke \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            seed=$SEED"
    else
        echo "python devops/skypilot/launch.py train \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            seed=$SEED"
    fi
}

# Test 1: 128 steps with varying BPTT horizons
for bptt in 64 128 256; do
    steps=128
    # Using the actual existing curriculum file
    curriculum="colortree_easy_${steps}step_${NUM_COLORS}colors"
    run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

    echo "Launching Test: 128 steps, BPTT=${bptt}"
    echo "  Run name: $run_name"
    echo "  Steps: ${steps}"
    echo "  Colors: ${NUM_COLORS}"
    echo "  BPTT Horizon: ${bptt}"
    echo "  Curriculum: ${curriculum}"
    echo "  Seed: ${SEED}"

    eval $(build_command "$run_name" "$curriculum" "$bptt")

    echo "---"
done

# Test 2: 64 steps with BPTT horizon of 128
steps=64
bptt=128
# For 64 steps, the curriculum file is just "colortree_easy_2colors"
curriculum="colortree_easy_${NUM_COLORS}colors"
run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

echo "Launching Test: 64 steps, BPTT=128"
echo "  Run name: $run_name"
echo "  Steps: ${steps}"
echo "  Colors: ${NUM_COLORS}"
echo "  BPTT Horizon: ${bptt}"
echo "  Curriculum: ${curriculum}"
echo "  Seed: ${SEED}"

eval $(build_command "$run_name" "$curriculum" "$bptt")

echo "---"

# Test 3: 32 steps with BPTT horizon of 64
steps=32
bptt=64
# Using the actual existing curriculum file
curriculum="colortree_easy_${steps}step_${NUM_COLORS}colors"
run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

echo "Launching Test: 32 steps, BPTT=64"
echo "  Run name: $run_name"
echo "  Steps: ${steps}"
echo "  Colors: ${NUM_COLORS}"
echo "  BPTT Horizon: ${bptt}"
echo "  Curriculum: ${curriculum}"
echo "  Seed: ${SEED}"

eval $(build_command "$run_name" "$curriculum" "$bptt")

echo "---"
echo "All 5 experiments launched!"
