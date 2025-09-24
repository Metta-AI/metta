#!/usr/bin/env python3
"""Test script to verify contrastive loss is enabled and logging works."""

import subprocess
import sys

def main():
    """Run a short training session with contrastive loss enabled."""

    # Create a test run with contrastive loss enabled
    cmd = [
        "uv", "run", "./tools/run.py", "experiments.recipes.arena.train",
        "run=contrastive_test",
        "trainer.total_timesteps=10000",  # Very short run for testing
        "trainer.losses.enable_contrastive=true"  # Enable contrastive loss
    ]

    print("Running training with contrastive loss enabled...")
    print("Command:", " ".join(cmd))
    print("\nLook for these messages in the output:")
    print("1. '[Contrastive Loss] Initialized with instance_name='contrastive'...'")
    print("2. '[Contrastive Loss] run_train called at step ...'")
    print("3. '[Contrastive Loss] Step X: Positive sim: ... Loss: ...'")
    print("4. Wandb metrics with 'contrastive_' prefix in the logs")
    print("=" * 60)

    # Run the command
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())