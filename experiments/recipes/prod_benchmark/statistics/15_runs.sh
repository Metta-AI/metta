#!/usr/bin/env bash
# Launches 15 training runs with different seeds on SkyPilot
#
# Usage:
#   ./run_icl_benchmark.sh <recipe>
#
# Example:
#   ./run_icl_benchmark.sh experiments.recipes.prod_benchmark.ICL.train

set -euo pipefail

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <recipe>"
    echo "Example: $0 experiments.recipes.prod_benchmark.ICL.train"
    exit 1
fi

# Configuration
RECIPE="$1"
BASE_RUN_NAME="benchmark_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

# Seeds for the 15 runs (using diverse seeds for better coverage)
SEEDS=(1072 2043 3014 4085 5156 6227 7298 8369 9440 10511 11582 12653 13724 14795 15866)

# Print configuration
echo "================================================================================"
echo "Benchmark Run Configuration"
echo "================================================================================"
echo "Recipe:           $RECIPE"
echo "Base run name:    $BASE_RUN_NAME"
echo "GPUs per run:     $NUM_GPUS"
echo "Number of runs:   ${#SEEDS[@]}"
echo "Seeds:            ${SEEDS[*]}"
echo "================================================================================"
echo ""

# Confirm before proceeding
read -p "Proceed with launching ${#SEEDS[@]} runs on SkyPilot? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Launch all runs
echo "Starting benchmark runs..."
echo ""

run_idx=1
for seed in "${SEEDS[@]}"; do
    run_name="${BASE_RUN_NAME}.seed${seed}"

    echo "[${run_idx}/${#SEEDS[@]}] Launching run: $run_name (seed=$seed)"

    ./devops/skypilot/launch.py \
        "$RECIPE" \
        "run=${run_name}" \
        "seed=${seed}" \
        --gpus="${NUM_GPUS}" \
        --heartbeat-timeout=3600 \
        --skip-git-check &

    run_idx=$((run_idx + 1))
    echo ""
done

# Wait for all background jobs to complete submission
wait

# Summary
echo "================================================================================"
echo "All ${#SEEDS[@]} runs have been submitted to SkyPilot"
echo "Monitor progress with: sky queue"
echo ""
echo "Run names:"
for seed in "${SEEDS[@]}"; do
    echo "  ${BASE_RUN_NAME}.seed${seed}"
done
echo "================================================================================"
