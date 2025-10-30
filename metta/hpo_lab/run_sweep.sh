#!/bin/bash
# HPO Lab Sweep Runner - silences common warnings

# Suppress PyTorch distributed warnings on macOS
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"

# Suppress Pydantic field warnings
export PYDANTIC_SILENCE_WARNINGS=1

# Run the sweep with clean output
echo "Starting HPO sweep for PPO on LunarLander..."
echo "This will run 100 trials by default. Use Ctrl+C to stop early."
echo ""

# Run the sweep
uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.ray_sweep sweep_name="${1:-ppo_lunarlander_sweep}"