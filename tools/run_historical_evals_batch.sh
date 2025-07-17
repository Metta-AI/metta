#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# List of runs to evaluate (deduplicated)
RUNS=(
    "jacke.sky_spiral_traditional_20250716_134205"
    "jacke.sky_spiral_traditional_20250716_112045"
    "jacke.sky_comprehensive_20250716_112035"
    "jacke.sky_raster_standard_20250716_145423"
    "jacke.sky_raster_standard_20250716_112030"
    "jacke.sky_spiral_traditional_20250716_145416"
    "jacke.sky_raster_standard_20250716_134153"
    "jacke.sky_raster_standard_20250716_110514"
    "jacke.sky_comprehensive_20250716_134110"
    "jacke.sky_spiral_traditional_20250716_110438"
)

# Configuration
EVAL_INTERVAL=1          # Evaluate every checkpoint
BATCH_SIZE=20            # Process 20 checkpoints at a time to avoid memory issues
MAX_WORKERS=4            # Max parallel workers per run
TIMEOUT=1800             # 30 minutes timeout per evaluation

echo "🚀 Starting historical evaluations for ${#RUNS[@]} runs"
echo "⚙️  Configuration:"
echo "   - Eval interval: every ${EVAL_INTERVAL} epochs"
echo "   - Batch size: ${BATCH_SIZE} checkpoints"
echo "   - Max workers: ${MAX_WORKERS}"
echo "   - Timeout: ${TIMEOUT}s per evaluation"
echo ""

# Track overall progress
total_runs=${#RUNS[@]}
successful_runs=0
failed_runs=0

# Process each run
for i in "${!RUNS[@]}"; do
    run_name="${RUNS[$i]}"
    run_num=$((i + 1))

    echo "📊 [$run_num/$total_runs] Processing run: $run_name"

    # Run the evaluation with error handling
    if ./tools/eval_historical_checkpoints_gpu.py \
        "$run_name" \
        --eval_interval "$EVAL_INTERVAL" \
        --batch_size "$BATCH_SIZE" \
        --max_workers "$MAX_WORKERS" \
        --timeout "$TIMEOUT"; then

        echo "✅ [$run_num/$total_runs] Successfully completed: $run_name"
        ((successful_runs++))
    else
        echo "❌ [$run_num/$total_runs] Failed: $run_name"
        ((failed_runs++))
    fi

    echo ""
done

# Final summary
echo "🎯 Batch Historical Evaluation Summary:"
echo "✅ Successful runs: $successful_runs"
echo "❌ Failed runs: $failed_runs"
echo "📊 Total runs: $total_runs"

if [[ $failed_runs -eq 0 ]]; then
    echo "🎉 All runs completed successfully!"
    exit 0
else
    echo "⚠️  Some runs failed. Check the logs above for details."
    exit 1
fi
