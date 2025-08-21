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
            echo ""
            echo "Examples:"
            echo "  $0                                    # Local training"
            echo "  $0 --distributed --gpus 4            # Single node, 4 GPUs (L4:4)"
            echo "  $0 --distributed --nodes 4           # 4 nodes, 1 GPU each"
            echo "  $0 --distributed --gpus 4 --nodes 2  # 2 nodes, 4 GPUs each"
            echo ""
            echo "Available GPU configurations:"
            echo "  - L4:4 (4 L4 GPUs on single node)"
            echo "  - L4:8 (8 L4 GPUs on single node)"
            echo "  - Multi-node: 1 L4 GPU per node"
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
        echo "   Using $NUM_NODES nodes with $NUM_GPUS GPU(s) each"

        # Validate configuration
        if [ "$NUM_GPUS" -gt 1 ]; then
            echo "⚠️  Warning: Multi-node with multiple GPUs per node may require custom configuration"
        fi

        # Multi-node training not yet implemented
        echo "❌ Error: Multi-node training not yet implemented"
        echo "   Please use single-node multi-GPU training instead"
        exit 1
    else
        echo "🔗 Launching single-node multi-GPU training..."
        echo "   Using 1 node with $NUM_GPUS GPU(s)"

        # Multi-GPU training not yet implemented
        echo "❌ Error: Multi-GPU training not yet implemented"
        echo "   Please use local training instead"
        exit 1
        else
            echo "❌ Error: Multi-GPU training not yet implemented"
            echo "   Please use local training instead"
            exit 1
        fi
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
