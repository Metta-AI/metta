#!/bin/bash
# Learning Progress Training Launch Script
# Usage: ./scripts/launch_learning_progress.sh [OPTIONS]

set -e

# Default values
NUM_EPISODES=1000
WANDB_PROJECT="metta"
WANDB_RUN_NAME="msb_lpdehyd_$(date +%s)"
DISTRIBUTED=false
NUM_GPUS=1
NUM_NODES=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --episodes NUM     Number of training episodes (default: 1000)"
            echo "  --project NAME     Wandb project name (default: metta)"
            echo "  --run-name NAME    Wandb run name (default: auto-generated)"
            echo "  --distributed      Run in distributed mode"
            echo "  --gpus NUM         Number of GPUs per node (default: 1)"
            echo "  --nodes NUM        Number of nodes (default: 1)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Launching Learning Progress Training"
echo "========================================"
echo "Episodes: $NUM_EPISODES"
echo "Wandb Project: $WANDB_PROJECT"
echo "Wandb Run Name: $WANDB_RUN_NAME"
echo "Distributed: $DISTRIBUTED"
echo "GPUs per node: $NUM_GPUS"
echo "Number of nodes: $NUM_NODES"
echo ""

if [ "$DISTRIBUTED" = true ]; then
    if [ "$NUM_NODES" -gt 1 ]; then
        echo "🔗 Launching multi-node distributed training..."
        sky launch configs/sweep/learning_progress_multi_node_launch.yaml \
            --env NUM_EPISODES=$NUM_EPISODES \
            --env WANDB_PROJECT=$WANDB_PROJECT \
            --env WANDB_RUN_NAME=$WANDB_RUN_NAME
    else
        echo "🔗 Launching single-node multi-GPU training..."
        sky launch configs/sweep/learning_progress_launch.yaml \
            --env NUM_EPISODES=$NUM_EPISODES \
            --env WANDB_PROJECT=$WANDB_PROJECT \
            --env WANDB_RUN_NAME=$WANDB_RUN_NAME
    fi
else
    echo "🏠 Running local training..."
    python experiments/user/arena_lp_test.py \
        --num_episodes $NUM_EPISODES \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $WANDB_RUN_NAME \
        --run_training
fi

echo "✅ Training completed!"
