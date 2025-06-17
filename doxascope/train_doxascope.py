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
    # Define paths
    # This assumes the script is run from the project root (e.g., the 'metta' repo root)
    project_root = Path.cwd()
    data_dir = project_root / "doxascope/data"
    raw_data_dir = data_dir / "raw_data"
    preprocessed_dir = data_dir / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

    print("🧠 Preprocessing Doxascope Data")
    print("=" * 50)
    preprocess_doxascope_data(raw_data_dir=raw_data_dir, preprocessed_dir=preprocessed_dir)
    print("=" * 50)

    print("🧠 Training Doxascope Neural Network")
    print("=" * 50)

    # Define path for the training data file
    preprocessed_data_path = preprocessed_dir / "training_data.npz"

    # Train the network
    trainer, test_accuracy = train_doxascope(data_path=preprocessed_data_path, output_dir=preprocessed_dir)

    if trainer is None:
        print("❌ Training script failed. Exiting.")
        sys.exit(1)

    print("=" * 50)
    print("🎉 Training Complete!")
    print(f"🎯 Final Test Accuracy: {test_accuracy:.2f}%")

    # Interpret results (updated thresholds for optimized network)
    if test_accuracy > 86:
        print("🌟 OUTSTANDING: Optimized spatial-temporal encoding!")
    elif test_accuracy > 85:
        print("✅ EXCELLENT: Strong spatial-temporal patterns found!")
    elif test_accuracy > 80:
        print("✅ GOOD: Clear spatial-temporal encoding detected!")
    elif test_accuracy > 25:
        print("✅ SUCCESS: Meaningful patterns detected!")
    else:
        print("❌ No clear spatial-temporal encoding found")

    print("\n📊 Check the generated plots for detailed analysis:")
    print(f"   - {preprocessed_dir / 'training_curves.png'}")
    print(f"   - {preprocessed_dir / 'confusion_matrix.png'}")
    print(f"   - {preprocessed_dir / 'attention_analysis.png'}")


if __name__ == "__main__":
    main()
