#!/bin/bash
set -euo pipefail

# Distributed Protein Optimizer Analysis Script
# Leverages multiple GPUs for parallel seed execution

# Default values
NUM_GPUS=${NUM_GPUS:-$(command -v nvidia-smi > /dev/null && nvidia-smi --list-gpus | wc -l || echo 1)}
NSEEDS=${NSEEDS:-32}
ACQUISITION=${ACQUISITION:-all}
RANDOMIZE=${RANDOMIZE:-both}
PROBLEM=${PROBLEM:-all}
STAGES=${STAGES:-standard}
OUTPUT_DIR=${OUTPUT_DIR:-./protein_analysis_distributed}
VERBOSE=${VERBOSE:-}

# WandB configuration
WANDB_ENABLED=${WANDB_ENABLED:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PROJECT=${WANDB_PROJECT:-metta}

# Display configuration
echo "[CONFIG] Protein Optimizer Distributed Analysis"
echo "  - GPUs available: $NUM_GPUS"
echo "  - Seeds per config: $NSEEDS"
echo "  - Acquisition functions: $ACQUISITION"
echo "  - Randomization: $RANDOMIZE"
echo "  - Problem(s): $PROBLEM"
echo "  - Stage config: $STAGES"
echo "  - Output directory: $OUTPUT_DIR"
if [ -n "$WANDB_ENABLED" ]; then
    echo "  - WandB: ENABLED (project: $WANDB_PROJECT, entity: ${WANDB_ENTITY:-default})"
fi
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build command arguments
ARGS="--nseeds=$NSEEDS"
ARGS="$ARGS --acquisition=$ACQUISITION"
ARGS="$ARGS --randomize-acquisition=$RANDOMIZE"
ARGS="$ARGS --problem=$PROBLEM"
ARGS="$ARGS --stages=$STAGES"
ARGS="$ARGS --out=$OUTPUT_DIR"

if [ -n "$VERBOSE" ]; then
    ARGS="$ARGS --verbose"
fi

# Add WandB flags if enabled
if [ -n "$WANDB_ENABLED" ]; then
    ARGS="$ARGS --wandb"
    if [ -n "$WANDB_ENTITY" ]; then
        ARGS="$ARGS --wandb-entity=$WANDB_ENTITY"
    fi
    ARGS="$ARGS --wandb-project=$WANDB_PROJECT"
fi

# Check if we have multiple GPUs and CUDA is available
if [ "$NUM_GPUS" -gt 1 ] && command -v nvidia-smi > /dev/null; then
    echo "[INFO] Running distributed on $NUM_GPUS GPUs using torchrun"
    echo ""
    
    # Use torchrun for distributed execution
    uv run torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr=localhost \
        --master_port=12355 \
        metta/sweep/protein_analysis_distributed.py \
        $ARGS
else
    echo "[INFO] Running on single device (CPU or single GPU)"
    echo ""
    
    # Single process execution
    uv run python metta/sweep/protein_analysis_distributed.py $ARGS
fi

echo ""
echo "[SUCCESS] Analysis completed. Results saved to: $OUTPUT_DIR"