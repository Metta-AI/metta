#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import argparse
from pathlib import Path

from .doxascope_network import train_doxascope


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train the doxascope neural network.")
    parser.add_argument("policy_name", type=str, help="Name of the policy to train on.")
    parser.add_argument(
        "num_future_timesteps",
        type=int,
        nargs="?",
        default=1,
        help="Number of future timesteps to predict.",
    )
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()

    print(f"Using policy: {args.policy_name}")
    if args.num_future_timesteps > 1:
        print(f"Predicting {args.num_future_timesteps} steps into the future.")

    # Define paths
    raw_data_dir = Path(f"doxascope/data/raw_data/{args.policy_name}")
    results_dir = Path(f"doxascope/data/results/{args.policy_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess data, then train the network
    results = train_doxascope(
        raw_data_dir=raw_data_dir,
        output_dir=results_dir,
        num_future_timesteps=args.num_future_timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    if results is None:
        print("❌ Training failed.")
        return

    print("\n" + "=" * 50)
    print("📊 Check the generated plots for detailed analysis:")
    for plot_path in results_dir.glob("*.png"):
        print(f"   - {plot_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
