#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .doxascope_analysis import run_analysis
from .doxascope_network import (
    DoxascopeNet,
    DoxascopeTrainer,
    prepare_data,
)


def train_doxascope(
    raw_data_dir: Path,
    output_dir: Path,
    batch_size=32,
    test_split=0.2,
    val_split=0.1,
    num_epochs=100,
    lr=0.0007,
    num_future_timesteps=1,
    num_past_timesteps=0,
):
    """
    Main training function for the Doxascope network.

    This function handles data preparation, model training, and result analysis.
    """
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data
    data_loaders, input_dim = prepare_data(
        raw_data_dir, output_dir, test_split, val_split, num_future_timesteps, num_past_timesteps
    )
    if not data_loaders:
        return None
    train_loader, val_loader, test_loader = data_loaders

    # Model configuration
    model_params = {
        "input_dim": input_dim,
        "num_future_timesteps": num_future_timesteps,
        "num_past_timesteps": num_past_timesteps,
        "hidden_dim": 512,
        "dropout_rate": 0.4,
        "activation_fn": "silu",
        "main_net_depth": 3,
        "processor_depth": 1,
        "skip_connection_weight": 0.1,
    }
    model = DoxascopeNet(**model_params).to(device)

    # Training
    trainer = DoxascopeTrainer(model, device)
    history, results, best_model_checkpoint = trainer.train(train_loader, val_loader, num_epochs=num_epochs, lr=lr)

    if output_dir and best_model_checkpoint:
        torch.save(best_model_checkpoint, output_dir / "best_model.pth")

    # Final analysis on test set
    if test_loader:
        _, test_acc_per_step, all_preds, all_targets = trainer._run_epoch(test_loader, is_training=False)
        results["test_acc_per_step"] = test_acc_per_step
        results["predictions"] = all_preds
        results["targets"] = all_targets

    if output_dir:
        run_analysis(history, results, output_dir)

    # Save analysis results
    if output_dir:
        analysis_data = {
            "policy_name": raw_data_dir.name,
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0,
            "avg_test_acc": (
                sum(results["test_acc_per_step"]) / len(results["test_acc_per_step"])
                if results["test_acc_per_step"]
                else 0
            ),
            "test_acc_per_step": results["test_acc_per_step"],
            "num_past_timesteps": num_past_timesteps,
            "num_future_timesteps": num_future_timesteps,
            "model_config": model.config,
        }
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(analysis_data, f, indent=4)

    end_time = time.time()
    print(f"\n✅ Training and analysis complete in {end_time - start_time:.2f} seconds.")
    if history["val_acc"]:
        print(f"📈 Best Validation Accuracy (avg): {max(history['val_acc']):.2f}%")

    # Final summary print
    if "test_acc_per_step" in results:
        test_acc_per_step = results["test_acc_per_step"]
        print("📈 Final Test Accuracies:")
        if test_acc_per_step:
            num_past = model.config.get("num_past_timesteps", 0)
            for i, acc in enumerate(test_acc_per_step):
                step = i - num_past + (1 if i >= num_past else 0)
                print(f"  - Step t{step:+.0f}: {acc:.2f}%")

    return (
        max(history["val_acc"]) if history["val_acc"] else 0,
        results.get("test_acc_per_step", [0])[0],
    )


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
    parser.add_argument(
        "--num-past-timesteps",
        type=int,
        default=0,
        help="Number of past timesteps to predict.",
    )
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()

    print(f"Using policy: {args.policy_name}")
    if args.num_future_timesteps > 1:
        print(f"Predicting {args.num_future_timesteps} steps into the future.")
    if args.num_past_timesteps > 0:
        print(f"Predicting {args.num_past_timesteps} steps into the past.")

    # Define paths
    raw_data_dir = Path(f"doxascope/data/raw_data/{args.policy_name}")
    results_dir = Path(f"doxascope/data/results/{args.policy_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess data, then train the network
    results = train_doxascope(
        raw_data_dir=raw_data_dir,
        output_dir=results_dir,
        num_future_timesteps=args.num_future_timesteps,
        num_past_timesteps=args.num_past_timesteps,
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
