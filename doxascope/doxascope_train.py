#!/usr/bin/env python3
"""
Train Doxascope Network

Simple script to train the doxascope neural network on LSTM memory vectors.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .doxascope_network import DoxascopeNet, DoxascopeTrainer, prepare_data


def run_training_pipeline(
    policy_name: str,
    raw_data_dir: Path,
    output_dir: Path,
    device: str,
    args: argparse.Namespace,
    is_baseline: bool = False,
):
    """Manages the full pipeline for a single training run (regular or baseline)."""
    print("\n" + "=" * 50)
    model_type = "Baseline" if is_baseline else "Doxascope"
    print(f"    Training {model_type} Model for {policy_name}    ")
    print("=" * 50)

    # Prepare data loaders
    data_result = prepare_data(
        raw_data_dir,
        output_dir,
        args.test_split,
        args.val_split,
        args.num_future_timesteps,
        args.num_past_timesteps,
        randomize_X=is_baseline,
    )

    if data_result[0] is None:
        print(f"Failed to create data loaders for {model_type} model. Skipping.")
        return

    train_loader, val_loader, test_loader, input_dim = data_result
    assert input_dim is not None, "Input dimension cannot be None"

    # Initialize model and trainer
    model = DoxascopeNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        num_future_timesteps=args.num_future_timesteps,
        num_past_timesteps=args.num_past_timesteps,
        activation_fn=args.activation_fn,
        main_net_depth=args.main_net_depth,
        processor_depth=args.processor_depth,
    ).to(device)

    trainer = DoxascopeTrainer(model, device=device)
    training_result = trainer.train(
        train_loader, val_loader, num_epochs=args.num_epochs, lr=args.learning_rate, patience=args.patience
    )

    if training_result is None:
        print(f"Training failed for {model_type} model (no validation improvements).")
        return

    # Save training history and best model
    suffix = "_baseline" if is_baseline else ""
    history_df = pd.DataFrame(training_result.history)
    history_df.to_csv(output_dir / f"training_history{suffix}.csv", index=False)
    print(f"{model_type} training history saved.")

    checkpoint_path = output_dir / f"best_model{suffix}.pth"
    torch.save(training_result.best_checkpoint, checkpoint_path)
    print(f"Best {model_type} model saved to {checkpoint_path}")

    # Final evaluation on the test set
    if test_loader:
        model.load_state_dict(training_result.best_checkpoint["state_dict"])
        test_loss, test_acc_per_step = trainer._run_epoch(test_loader, is_training=False)
        avg_test_acc = sum(test_acc_per_step) / len(test_acc_per_step) if test_acc_per_step else 0

        print(f"\nFinal {model_type} Test Set Evaluation:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Average Test Accuracy: {avg_test_acc:.2f}%")

        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_accuracy": avg_test_acc,
            "test_accuracy_per_step": test_acc_per_step,
            "timesteps": model.head_timesteps,
        }
        with open(output_dir / f"test_results{suffix}.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"{model_type} test results saved.")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a DoxascopeNet model.")
    parser.add_argument("policy_name", type=str, help="Name of the policy to train on.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("train_dir/doxascope/raw_data"),
        help="Directory containing the raw doxascope data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory to save the preprocessed data and model checkpoints.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Unique name for this training run.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Proportion of data to use for the test set.")
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Proportion of data to use for the validation set."
    )
    parser.add_argument("--num-future-timesteps", type=int, default=1, help="Number of future timesteps to predict.")
    parser.add_argument("--num-past-timesteps", type=int, default=0, help="Number of past timesteps to predict.")

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda')")
    parser.add_argument(
        "--train-random-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train a baseline model with random memory vectors for comparison.",
    )

    # Add arguments for model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for the model.")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout rate for the model.")
    parser.add_argument("--activation_fn", type=str, default="silu", help="Activation function for the model.")
    parser.add_argument("--main_net_depth", type=int, default=3, help="Depth of the main network.")
    parser.add_argument("--processor_depth", type=int, default=1, help="Depth of the state processors.")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Prepare data and output directories
    policy_data_dir = args.raw_data_dir / args.policy_name
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / args.policy_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # Run the main training pipeline
    run_training_pipeline(args.policy_name, policy_data_dir, output_dir, device, args, is_baseline=False)

    # Run the baseline training pipeline if requested
    if args.train_random_baseline:
        run_training_pipeline(args.policy_name, policy_data_dir, output_dir, device, args, is_baseline=True)


if __name__ == "__main__":
    main()
