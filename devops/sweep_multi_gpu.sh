#!/bin/bash
# sweep_multi_gpu.sh - Launch multiple sweep runs on different GPUs
# Example script showing how to run independent sweep runs on separate GPUs

set -e

# Usage: ./sweep_multi_gpu.sh <sweep_name> <gpus_per_run> <num_runs> [additional_args]
# Example: ./sweep_multi_gpu.sh my_sweep 2 4  # 4 runs, each using 2 GPUs

SWEEP_NAME="${1:-test_sweep}"
GPUS_PER_RUN="${2:-1}"
NUM_RUNS="${3:-4}"
TOTAL_GPUS="${4:-8}"  # Total GPUs available on the system

echo "[INFO] Launching $NUM_RUNS sweep runs for: $SWEEP_NAME"
echo "[INFO] Each run will use $GPUS_PER_RUN GPU(s)"
echo "[INFO] Total GPUs available: $TOTAL_GPUS"

# Calculate GPU assignments
gpu_offset=0
for run_idx in $(seq 0 $((NUM_RUNS - 1))); do
    # Create GPU list for this run
    gpu_list=""
    for gpu_num in $(seq 0 $((GPUS_PER_RUN - 1))); do
        gpu_id=$((gpu_offset + gpu_num))
        if [ -z "$gpu_list" ]; then
            gpu_list="$gpu_id"
        else
            gpu_list="${gpu_list},${gpu_id}"
        fi
    done

    echo "[INFO] Starting sweep run $run_idx on GPUs: $gpu_list"

    # Launch in background with GPU list
    ./devops/sweep.sh run="${SWEEP_NAME}.run${run_idx}" \
        gpu_list=$gpu_list \
        "${@:5}" &

    # Update offset for next run
    gpu_offset=$((gpu_offset + GPUS_PER_RUN))

    # Small delay to avoid race conditions
    sleep 2
done

echo "[INFO] All $NUM_RUNS sweep runs launched"
echo "[INFO] Use 'ps aux | grep sweep_rollout' to monitor progress"
echo "[INFO] Logs will be in separate directories for each run"

# Wait for all background jobs
wait
echo "[INFO] All sweep runs completed"
