#!/bin/bash

# ColorTree BPTT Horizon Experiments with sqrt(N) Learning Rate Scaling
# Tests different BPTT horizons with learning rate scaled by 1/sqrt(bptt_horizon)
# This sqrt(N) scaling approach maintains stable learning across different sequence lengths
# Note: Uses existing curriculum files and can optionally use user config

# Generate random seed if not provided
SEED=${1:-$RANDOM}
USE_USER_CONFIG=${2:-"no"}  # Pass "user" as second arg to include user config
echo "=== ColorTree BPTT sqrt(N) Learning Rate Scaling Experiments ==="
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
BASE_LR=0.0008  # Base learning rate (from jacke.yaml)

# Build base command with optional user config
build_command() {
    local run_name=$1
    local curriculum=$2
    local bptt=$3

    # Calculate appropriate batch size based on BPTT horizon
    # Need batch_size >= num_agents * bptt_horizon
    # Assuming 8192 agents from the error message
    local num_agents=8192
    local min_batch_size=$((num_agents * bptt))
    # Round up to next power of 2 for efficiency
    local batch_size=524288  # Default
    if [ $bptt -eq 128 ]; then
        batch_size=1048576  # 1M
    elif [ $bptt -eq 256 ]; then
        batch_size=2097152  # 2M
    fi

    # Calculate learning rate with 1/sqrt(N) scaling
    # Using bc for floating point arithmetic
    # LR = constant / sqrt(bptt), where constant is chosen so LR=BASE_LR at BPTT=128
    # constant = BASE_LR * sqrt(128) = BASE_LR * 11.3137085
    local lr=$(echo "scale=8; $BASE_LR * sqrt(128) / sqrt($bptt)" | bc)

    if [ "$USE_USER_CONFIG" = "user" ]; then
        echo "python devops/skypilot/launch.py train user=jacke \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            trainer.batch_size=${batch_size} \
            trainer.optimizer.learning_rate=${lr} \
            seed=$SEED"
    else
        echo "python devops/skypilot/launch.py train \
            run=$run_name \
            trainer.curriculum=env/mettagrid/curriculum/${curriculum} \
            sim=colortree_nosim \
            trainer.bptt_horizon=${bptt} \
            trainer.batch_size=${batch_size} \
            trainer.optimizer.learning_rate=${lr} \
            seed=$SEED"
    fi
}

# Test 1: 128 steps with varying BPTT horizons
for bptt in 64 128 256; do
    steps=128
    # Using the actual existing curriculum file
    curriculum="colortree_easy_${steps}step_${NUM_COLORS}colors"
    run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_sqrtN_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

    # Calculate LR for display
    lr=$(echo "scale=8; $BASE_LR * sqrt(128) / sqrt($bptt)" | bc)

    echo "Launching Test: 128 steps, BPTT=${bptt}, 1/sqrt(N) LR scaling"
    echo "  Run name: $run_name"
    echo "  Steps: ${steps}"
    echo "  Colors: ${NUM_COLORS}"
    echo "  BPTT Horizon: ${bptt}"
    echo "  Learning Rate: ${lr} (${BASE_LR} * sqrt(128) / sqrt(${bptt}))"
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
run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_sqrtN_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

# Calculate LR for display
lr=$(echo "scale=8; $BASE_LR * 8 / sqrt($bptt)" | bc)

echo "Launching Test: 64 steps, BPTT=128, 1/sqrt(N) LR scaling"
echo "  Run name: $run_name"
echo "  Steps: ${steps}"
echo "  Colors: ${NUM_COLORS}"
echo "  BPTT Horizon: ${bptt}"
echo "  Learning Rate: ${lr} (${BASE_LR} * 8 / sqrt(${bptt}))"
echo "  Curriculum: ${curriculum}"
echo "  Seed: ${SEED}"

eval $(build_command "$run_name" "$curriculum" "$bptt")

echo "---"

# Test 3: 32 steps with BPTT horizon of 64
steps=32
bptt=64
# Using the actual existing curriculum file
curriculum="colortree_easy_${steps}step_${NUM_COLORS}colors"
run_name="${USER}.colortree_${steps}step_${NUM_COLORS}colors_bptt${bptt}_sqrtN_seed${SEED}.$(date +%Y%m%d_%H%M%S)"

# Calculate LR for display
lr=$(echo "scale=8; $BASE_LR * 8 / sqrt($bptt)" | bc)

echo "Launching Test: 32 steps, BPTT=64, 1/sqrt(N) LR scaling"
echo "  Run name: $run_name"
echo "  Steps: ${steps}"
echo "  Colors: ${NUM_COLORS}"
echo "  BPTT Horizon: ${bptt}"
echo "  Learning Rate: ${lr} (${BASE_LR} * 8 / sqrt(${bptt}))"
echo "  Curriculum: ${curriculum}"
echo "  Seed: ${SEED}"

eval $(build_command "$run_name" "$curriculum" "$bptt")

echo "---"
echo ""
echo "=== All 5 experiments with 1/sqrt(N) LR scaling launched! ==="
echo "Learning rates used (normalized to BPTT=128):"
echo "  BPTT=64:  $(echo "scale=8; $BASE_LR * sqrt(128) / sqrt(64)" | bc) (higher than baseline)"
echo "  BPTT=128: $(echo "scale=8; $BASE_LR * sqrt(128) / sqrt(128)" | bc) = ${BASE_LR} (baseline)"
echo "  BPTT=256: $(echo "scale=8; $BASE_LR * sqrt(128) / sqrt(256)" | bc) (lower than baseline)"
