#!/usr/bin/env bash
# Launches 15 training runs with different seeds on SkyPilot
#
# Usage:
#   ./15_runs.sh <recipe> [--uniform-seed SEED]
#
# Examples:
#   ./15_runs.sh experiments.recipes.prod_benchmark.ICL.train
#   ./15_runs.sh experiments.recipes.prod_benchmark.abes.train --uniform-seed 63

set -euo pipefail

# Parse arguments
RECIPE=""
UNIFORM_SEED=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --uniform-seed)
            if [[ $# -lt 2 ]]; then
                echo "Error: --uniform-seed requires a seed value" >&2
                exit 1
            fi
            UNIFORM_SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <recipe> [--uniform-seed SEED]"
            exit 0
            ;;
        *)
            if [[ -z "$RECIPE" ]]; then
                RECIPE="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$RECIPE" ]]; then
    echo "Usage: $0 <recipe> [--uniform-seed SEED]" >&2
    exit 1
fi

# Include recipe identifier in the run name so runs are easy to group in W&B
# Replace dots with underscores for readability, then drop the literal word 'recipes'
RECIPE_SLUG="${RECIPE//./_}"
RECIPE_SLUG="${RECIPE_SLUG//recipes/}"
BASE_RUN_NAME="benchmark_${RECIPE_SLUG}_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

if [[ -n "$UNIFORM_SEED" ]]; then
    SEEDS=()
    for _ in {1..15}; do
        SEEDS+=("$UNIFORM_SEED")
    done
    echo "Configured uniform seed: $UNIFORM_SEED (15 runs)"
else
    # Default: 15 unique seeds for power analysis
    SEEDS=(1072 2043 3014 4085 5156 6227 7298 8369 9440 10511 11582 12653 13724 14795 15866)
fi

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
