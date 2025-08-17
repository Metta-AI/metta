#!/bin/bash
# Test script for AGaLiTe fix - matching the working configuration

echo "=== Testing AGaLiTe Fix ==="
echo "This test matches the working configuration from commit 350a32ad"
echo ""

# Set test ID
export TEST_ID=agalite_fix_$(date +%Y%m%d_%H%M%S)
echo "Test ID: $TEST_ID"
echo ""

# Test commands for both AGaLiTe variants
echo "Testing AGaLiTe (fast mode default):"
echo "------------------------------------"
echo "Command: uv run ./tools/train.py py_agent=agalite trainer.num_workers=2 trainer.total_timesteps=1000 wandb=off trainer.simulation.skip_git_check=true run=test_agalite_$TEST_ID"
echo ""

echo "Testing AGaLiTe Improved (experimental):"
echo "----------------------------------------"
echo "Command: uv run ./tools/train.py py_agent=agalite_improved trainer.num_workers=2 trainer.total_timesteps=1000 wandb=off trainer.simulation.skip_git_check=true run=test_agalite_improved_$TEST_ID"
echo ""

echo "To run a quick test (100 timesteps), use:"
echo "HYDRA_FULL_ERROR=1 uv run ./tools/train.py py_agent=agalite trainer.num_workers=2 trainer.total_timesteps=100 wandb=off trainer.simulation.skip_git_check=true"
echo ""

echo "To monitor for NaN issues, look for:"
echo "- WARNING messages about invalid logits or hidden representations"
echo "- Zero rewards throughout training (indicates model not learning)"
echo "- CUDA assertion errors"
echo ""

echo "Expected behavior after fix:"
echo "- No NaN warnings"
echo "- Non-zero rewards should appear within first 1000 timesteps"
echo "- Stable training without crashes"