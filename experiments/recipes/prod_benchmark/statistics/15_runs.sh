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

ensure_site_packages_visible() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        return
    fi
    if ! command -v chflags >/dev/null 2>&1; then
        return
    fi
    local repo_root
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
    if [[ ! -d "$repo_root/.venv" ]]; then
        return
    fi
    shopt -s nullglob
    for site_dir in "$repo_root"/.venv/lib/python*/site-packages; do
        if [[ -d "$site_dir" ]]; then
            chflags -R nohidden "$site_dir" 2>/dev/null || true
        fi
    done
    shopt -u nullglob
}

# Ensure editable packages like gitta stay importable when macOS hides site-packages
ensure_site_packages_visible

# Parse arguments
RECIPE=""
UNIFORM_SEED=""
EXTRA_TOOL_ARGS=()
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
                EXTRA_TOOL_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

if [[ -z "$RECIPE" ]]; then
    echo "Usage: $0 <recipe> [--uniform-seed SEED]" >&2
    exit 1
fi

# Extract just the recipe name (e.g., "navigation" from "experiments.recipes.prod_benchmark.navigation.train")
# Format: {recipe_name}.{date}.seed{seed}.run{number}
RECIPE_PATH="${RECIPE%.*}"  # Remove the command part (e.g., .train)
RECIPE_NAME="${RECIPE_PATH##*.}"  # Get the last component (e.g., navigation)
BASE_RUN_NAME="${RECIPE_NAME}.$(date +%Y%m%d)"
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
RUN_NAMES=()
for seed in "${SEEDS[@]}"; do
    run_suffix="run$(printf '%02d' "$run_idx")"
    run_name="${BASE_RUN_NAME}.seed${seed}.${run_suffix}"
    RUN_NAMES+=("$run_name")

    echo "[${run_idx}/${#SEEDS[@]}] Launching run: $run_name (seed=$seed)"

    launch_cmd=(
        ./devops/skypilot/launch.py
        "$RECIPE"
        "run=${run_name}"
        "seed=${seed}"
    )

    if [[ ${#EXTRA_TOOL_ARGS[@]} -gt 0 ]]; then
        launch_cmd+=("${EXTRA_TOOL_ARGS[@]}")
    fi

    launch_cmd+=(
        --gpus="${NUM_GPUS}"
        --heartbeat-timeout=3600
        --skip-git-check
    )

    "${launch_cmd[@]}" &

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
for run_name in "${RUN_NAMES[@]}"; do
    echo "  $run_name"
done
echo "================================================================================"
