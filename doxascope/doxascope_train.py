#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import sys
from pathlib import Path

from .doxascope_network import train_doxascope


def main():
    """Main CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python -m doxascope.doxascope_train <policy_name> [num_future_timesteps]")
        sys.exit(1)

    policy_name = sys.argv[1]
    num_future_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Using policy: {policy_name}")
    if num_future_timesteps > 1:
        print(f"Predicting {num_future_timesteps} steps into the future.")

    # Define paths
    raw_data_dir = Path(f"doxascope/data/raw_data/{policy_name}")
    results_dir = Path(f"doxascope/data/results/{policy_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess data, then train the network
    trainer, test_accuracy = train_doxascope(
        raw_data_dir=raw_data_dir,
        output_dir=results_dir,
        num_future_timesteps=num_future_timesteps,
    )

    if trainer is None:
        print("❌ Training failed.")
        return

    print("\n" + "=" * 50)
    print("📊 Check the generated plots for detailed analysis:")
    for plot_path in results_dir.glob("*.png"):
        print(f"   - {plot_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
