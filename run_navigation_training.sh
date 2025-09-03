#!/bin/bash

# Navigation Training Script
# Runs navigation training with both learning progress and random curricula

set -e  # Exit on any error

echo "🚀 Starting Navigation Training Comparison"
echo "=========================================="

# Configuration
WANDB_GROUP="navigation_learning_progress"

echo "📊 Training Configuration:"
echo "   WandB group: $WANDB_GROUP"
echo "   Using default trainer configuration"
echo ""

# Function to run training
run_training() {
    local curriculum_type=$1
    local use_lp=$2
    local run_name="navigation_${curriculum_type}_$(date +%m%d_%H%M%S)"

    echo "🎯 Running $curriculum_type curriculum training..."
    echo "   Run name: $run_name"
    echo "   Learning progress: $use_lp"
    echo ""

    uv run ./tools/run.py experiments.recipes.navigation.train \
        --args \
        run="$run_name" \
        use_learning_progress="$use_lp" \
        wandb_group="$WANDB_GROUP"

    echo "✅ $curriculum_type training completed"
    echo ""
}

# Run learning progress curriculum training
echo "🔄 Starting Learning Progress Curriculum Training..."
run_training "learning_progress" "true"

# Wait a moment between runs
sleep 5

# Run random curriculum training
echo "🔄 Starting Random Curriculum Training..."
run_training "random" "false"

echo "🎉 All training runs completed!"
echo "📈 Check WandB group '$WANDB_GROUP' for results comparison"
