#!/usr/bin/env bash
# Launches 15 ICL training runs with different seeds for variance analysis
#
# Usage:
#   ./run_icl_benchmark.sh [OPTIONS]
#
# Options:
#   --architecture <arch>     Architecture to use: vit_reset (default) or trxl
#   --curriculum <style>      Curriculum style (default: terrain_2)
#   --base-run-name <name>    Base name for runs (default: icl_benchmark_TIMESTAMP)
#   --skypilot                Launch on SkyPilot (default: local)
#   --gpus <n>                Number of GPUs for SkyPilot (default: 4)
#   --dry-run                 Print commands without executing
#   --help                    Show this help message

set -euo pipefail

# Default configuration
ARCHITECTURE="vit_reset"
CURRICULUM="terrain_2"
BASE_RUN_NAME="icl_benchmark_$(date +%Y%m%d_%H%M%S)"
USE_SKYPILOT=true
NUM_GPUS=4
DRY_RUN=false

# Seeds for the 15 runs (using diverse seeds for better coverage)
SEEDS=(1072 2043 3014 4085 5156 6227 7298 8369 9440 10511 11582 12653 13724 14795 15866)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --curriculum)
            CURRICULUM="$2"
            shift 2
            ;;
        --base-run-name)
            BASE_RUN_NAME="$2"
            shift 2
            ;;
        --skypilot)
            USE_SKYPILOT=true
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate architecture
if [[ "$ARCHITECTURE" != "vit_reset" && "$ARCHITECTURE" != "trxl" ]]; then
    echo "Error: Architecture must be 'vit_reset' or 'trxl'"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "ICL Benchmark Run Configuration"
echo "================================================================================"
echo "Architecture:     $ARCHITECTURE"
echo "Curriculum:       $CURRICULUM"
echo "Base run name:    $BASE_RUN_NAME"
echo "Launch method:    $([ "$USE_SKYPILOT" = true ] && echo "SkyPilot (${NUM_GPUS} GPUs)" || echo "Local")"
echo "Number of runs:   ${#SEEDS[@]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Dry run:          $DRY_RUN"
echo "================================================================================"
echo ""

# Function to launch a single run
launch_run() {
    local seed=$1
    local run_idx=$2
    local run_name="${BASE_RUN_NAME}.${ARCHITECTURE}.seed${seed}"

    echo "[${run_idx}/${#SEEDS[@]}] Launching run: $run_name (seed=$seed)"

    if [ "$USE_SKYPILOT" = true ]; then
        # Launch on SkyPilot
        cmd=(
            "./devops/skypilot/launch.py"
            "experiments.recipes.prod_benchmark.ICL.train"
            "run=${run_name}"
            "curriculum_style=${CURRICULUM}"
            "architecture=${ARCHITECTURE}"
            "seed=${seed}"
            "--gpus=${NUM_GPUS}"
            "--heartbeat-timeout=3600"
            "--skip-git-check"
        )
    else
        # Launch locally using tools/run.py
        cmd=(
            "uv" "run" "./tools/run.py"
            "experiments.recipes.prod_benchmark.ICL.train"
            "run=${run_name}"
            "curriculum_style=${CURRICULUM}"
            "architecture=${ARCHITECTURE}"
            "seed=${seed}"
        )
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would execute: ${cmd[*]}"
    else
        echo "  Executing: ${cmd[*]}"
        "${cmd[@]}" &

        # If running locally, add a small delay between launches to avoid resource contention
        if [ "$USE_SKYPILOT" = false ]; then
            sleep 5
        fi
    fi

    echo ""
}

# Confirm before proceeding (unless dry run)
if [ "$DRY_RUN" = false ]; then
    read -p "Proceed with launching ${#SEEDS[@]} runs? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Launch all runs
echo "Starting benchmark runs..."
echo ""

run_idx=1
for seed in "${SEEDS[@]}"; do
    launch_run "$seed" "$run_idx"
    run_idx=$((run_idx + 1))
done

# Summary
echo "================================================================================"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE"
    echo "No runs were actually launched. Remove --dry-run to execute."
else
    if [ "$USE_SKYPILOT" = true ]; then
        echo "All ${#SEEDS[@]} runs have been submitted to SkyPilot"
        echo "Monitor progress with: sky queue"
    else
        echo "All ${#SEEDS[@]} runs have been launched locally in background"
        echo "Monitor progress with: ps aux | grep train"
        echo "Logs will be in: train_dir/<run_name>/"
    fi

    echo ""
    echo "After all runs complete, analyze variance with:"
    echo "  python experiments/recipes/prod_benchmark/variance.py \\"
    for seed in "${SEEDS[@]}"; do
        echo "    ${BASE_RUN_NAME}.${ARCHITECTURE}.seed${seed} \\"
    done | sed '$ s/ \\$//'
fi
echo "================================================================================"
