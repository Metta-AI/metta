#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import sys
from pathlib import Path

from .doxascope_data import preprocess_doxascope_data
from .doxascope_network import train_doxascope


def main():
    """Main function to run the training script."""
    # Check for policy name argument
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a policy name as an argument.")
        print("   Usage: python -m doxascope.doxascope_train <policy_name>")
        sys.exit(1)

    policy_name = sys.argv[1]

    # Define paths
    project_root = Path.cwd()
    data_dir = project_root / "doxascope/data"
    raw_data_dir = data_dir / "raw_data" / policy_name
    results_dir = data_dir / "results" / policy_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using policy: {policy_name}")
    print("Preprocessing Doxascope Data")
    print("=" * 50)
    preprocess_doxascope_data(raw_data_dir=raw_data_dir, preprocessed_dir=results_dir)
    print("Training Doxascope Neural Network")
    print("=" * 50)

    # Define path for the training data file
    preprocessed_data_path = results_dir / "training_data.npz"

    # Train the network
    trainer, test_accuracy = train_doxascope(data_path=preprocessed_data_path, output_dir=results_dir)

    if trainer is None:
        print("❌ Training script failed. Exiting.")
        sys.exit(1)

    print("=" * 50)
    print("\n📊 Check the generated plots for detailed analysis:")
    print(f"   - {results_dir / 'training_curves.png'}")
    print(f"   - {results_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
