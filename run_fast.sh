#!/bin/bash

# Fast training script for significant speedup
# This script runs the optimized training with key performance improvements

set -e

echo "🚀 Starting FAST training for significant speedup..."

# Set environment variables for performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Set Python optimizations
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0

# Run the fast training script
echo "📊 Using fast configuration..."
echo "🔧 Batch size: 131,072 (128K)"
echo "🔧 Minibatch size: 16,384 (16K)"
echo "🔧 Workers: 8"
echo "🔧 Environments: 256"
echo "🔧 Async factor: 4"

# Run with the fast script
python run_fast.py

echo "✅ Fast training complete!"
