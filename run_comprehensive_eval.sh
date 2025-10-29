#!/bin/bash
# Comprehensive evaluation script using cogames eval command

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="eval_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/evaluation.log"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "COMPREHENSIVE EVALUATION - $TIMESTAMP" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run evaluation on all machina_eval missions
echo "Running evaluation on all machina_eval missions..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

uv run cogames eval -m "machina_eval.*" -p scripted -e 3 -s 1000 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "EVALUATION COMPLETE" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

