#!/bin/bash

# Ultra-fast training script for 10-20x speedup
# This script runs the optimized training with maximum performance settings

set -e

echo "🚀 Starting ULTRA-FAST training for 10-20x speedup..."

# Set environment variables for maximum performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# Enable PyTorch optimizations
export TORCH_COMPILE_MODE=reduce-overhead
# export TORCH_LOGS=off  # This setting is invalid, removing it

# Set Python optimizations
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0

# Run the optimized training script
echo "📊 Using ultra-fast configuration..."
echo "🔧 Batch size: 262,144 (256K)"
echo "🔧 Minibatch size: 32,768 (32K)"
echo "🔧 Workers: 16"
echo "🔧 Environments: 512"
echo "🔧 Async factor: 8"

# Run with the optimized script
python run_optimized.py

echo "✅ Ultra-fast training complete!"
