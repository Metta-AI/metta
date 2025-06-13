#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from doxascope import train_doxascope

if __name__ == "__main__":
    print("🧠 Training Doxascope Neural Network")
    print("=" * 50)

    # Train the network with optimized parameters
    trainer, test_accuracy = train_doxascope(data_path="../data/preprocessed_data/training_data.npz")

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
    print("   - doxascope/data/preprocessed_data/training_curves.png")
    print("   - doxascope/data/preprocessed_data/confusion_matrix.png")
    print("   - doxascope/data/preprocessed_data/attention_analysis.png")
