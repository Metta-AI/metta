#!/usr/bin/env python3
"""
Example: Training with VAPOR (Variational Policy Optimization)

This script demonstrates how to train using VAPOR instead of standard PPO.
VAPOR reformulates RL as variational inference over state-action optimality,
often providing better exploration and stability.

Usage:
    python examples/train_with_vapor.py

Or with custom parameters:
    python examples/train_with_vapor.py run=vapor_experiment +hardware=macbook
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run VAPOR training with appropriate configuration."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Build the training command
    cmd = [
        sys.executable,
        str(project_root / "tools" / "train.py"),
        # Use the VAPOR trainer configuration
        "trainer=vapor",
        # Set a descriptive run name
        "run=vapor_experiment",
        # Use macbook hardware config for local development
        "+hardware=macbook",
        # Disable wandb for local testing (enable if you want logging)
        "wandb=off",
        # Set a smaller number of workers for local testing
        "trainer.num_workers=2",
        # Optional: reduce total timesteps for faster testing
        "trainer.total_timesteps=10_000_000",
        # Optional: increase logging frequency to see VAPOR metrics
        "trainer.grad_mean_variance_interval=100",
    ]

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    print("Starting VAPOR training with command:")
    print(" ".join(cmd))
    print("\nVAPOR Configuration:")
    print("- Algorithm: VAPOR (Variational Policy Optimization)")
    print("- Beta: 1.0 (will anneal to 0.1 during training)")
    print("- Beta Schedule: Linear annealing")
    print("- Exploration: Uncertainty-based exploration bonus")
    print("- Importance Weighting: Enabled")
    print("\nExpected improvements over PPO:")
    print("- Better exploration through posterior uncertainty")
    print("- More stable policy updates via variational inference")
    print("- Principled handling of state-action optimality")
    print("\nMonitor these VAPOR-specific metrics in logs:")
    print("- vapor_policy_loss: VAPOR policy gradient loss")
    print("- vapor_kl_penalty: KL regularization penalty")
    print("- vapor_exploration_bonus: Uncertainty-based exploration")
    print("- vapor_posterior_entropy: Entropy of optimality posterior")
    print("- vapor_beta: Current beta (temperature) value")
    print()

    # Run the training command
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
