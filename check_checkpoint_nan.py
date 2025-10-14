#!/usr/bin/env python3
"""Quick script to check if a checkpoint contains NaN values."""

import sys

import torch

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint_nan.py <checkpoint_path.pt>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

print(f"Loading checkpoint: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check different possible structures
    state_dict = None
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, "state_dict"):
        # It's a model object
        state_dict = checkpoint.state_dict()
    else:
        print(f"Unknown checkpoint format: {type(checkpoint)}")
        sys.exit(1)

    nan_found = False
    inf_found = False

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            has_nan = torch.isnan(param).any()
            has_inf = torch.isinf(param).any()

            if has_nan or has_inf:
                nan_count = torch.isnan(param).sum().item() if has_nan else 0
                inf_count = torch.isinf(param).sum().item() if has_inf else 0
                print(f"⚠️  {name}: shape={param.shape}, NaN={nan_count}, Inf={inf_count}")
                nan_found = nan_found or has_nan
                inf_found = inf_found or has_inf

    if not nan_found and not inf_found:
        print("✓ Checkpoint is clean - no NaN or Inf values found")
    else:
        print("\n🔴 Checkpoint contains corrupted values!")
        print("   You need to train from an earlier checkpoint or from scratch.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)
