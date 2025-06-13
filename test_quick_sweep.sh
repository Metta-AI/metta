#!/bin/bash
# Test script for quick sweep - designed to complete in ~30 minutes

echo "🚀 Starting Quick Sweep Test (estimated completion: ~30 minutes)"
echo "=================================================="

# Run the quick sweep with Mac hardware configuration
./devops/sweep.sh \
  run=quick_sweep_test \
  ++sweep_params=sweep/quick_sweep \
  +hardware=macbook \
  +user=axel \
  trainer.num_workers=1 \
  trainer.total_timesteps=50000 \
  wandb.enabled=true

echo "=================================================="
echo "✅ Quick sweep test completed!"
echo "📊 Check results at: train_dir/sweep/quick_sweep_test/"
echo "🔗 WandB dashboard: https://wandb.ai/metta-research/metta"
