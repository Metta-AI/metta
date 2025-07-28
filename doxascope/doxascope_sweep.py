#!/usr/bin/env python3
"""
Doxascope Hyperparameter Sweep

This script runs a hyperparameter sweep for the Doxascope network to find
the optimal configuration for predicting agent actions from memory vectors.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from .doxascope_network import DoxascopeNet, DoxascopeTrainer, prepare_data


def get_search_space(sweep_type: str) -> Dict[str, list]:
    """Defines the search space for hyperparameters or architecture."""
    if sweep_type == "arch":
        print("🔬 Using architectural search space.")
        return {
            "hidden_dim": [512],
            "dropout_rate": [0.4],
            "lr": [0.0007],
            "activation_fn": ["gelu", "relu", "silu"],
            "main_net_depth": [1, 2, 3],
            "processor_depth": [1, 2],
        }
    print("⚙️ Using hyperparameter search space.")
    return {
        "hidden_dim": [128, 256, 512],
        "dropout_rate": [0.2, 0.4, 0.6],
        "lr": [0.0001, 0.0005, 0.001],
        "activation_fn": ["silu"],
        "main_net_depth": [3],
        "processor_depth": [1],
    }


def sample_config(search_space: Dict[str, list]) -> Dict[str, Any]:
    """Samples a single random configuration from the search space."""
    return {param: random.choice(values) for param, values in search_space.items()}


def run_sweep_trial(
    trial_idx: int,
    num_trials: int,
    config: Dict[str, Any],
    args: argparse.Namespace,
    device: str,
    data_loaders: tuple,
) -> Dict[str, Any]:
    """Runs a single trial of the sweep."""
    print("\n" + "=" * 50)
    print(f"  TRIAL {trial_idx + 1}/{num_trials} | Config: {config}")
    print("=" * 50)

    train_loader, val_loader, test_loader, input_dim = data_loaders
    assert input_dim is not None, "Input dimension cannot be None for sweep trial."

    model_params = {
        "input_dim": input_dim,
        "num_future_timesteps": args.num_future_timesteps,
        "num_past_timesteps": args.num_past_timesteps,
        **config,
    }
    model = DoxascopeNet(**model_params).to(device)
    trainer = DoxascopeTrainer(model, device=device)

    start_time = time.time()
    training_result = trainer.train(
        train_loader, val_loader, num_epochs=args.max_epochs, lr=config.get("lr", 0.001), patience=args.patience
    )
    train_time = time.time() - start_time

    if training_result is None:
        print("   ❌ Trial failed (no validation improvements).")
        return {"config": config, "success": False, "reason": "early_stopping_no_improvement"}

    # Evaluate on the test set
    model.load_state_dict(training_result.best_checkpoint["state_dict"])
    _, test_acc_per_step = trainer._run_epoch(test_loader, is_training=False)
    test_acc_avg = sum(test_acc_per_step) / len(test_acc_per_step) if test_acc_per_step else 0

    print(f"   ✅ Val Acc: {training_result.final_val_acc:.2f}%, Test Acc (avg): {test_acc_avg:.2f}%")
    return {
        "config": config,
        "success": True,
        "best_val_acc": training_result.final_val_acc,
        "test_acc_avg": test_acc_avg,
        "test_acc_per_step": test_acc_per_step,
        "train_time": train_time,
        "history": training_result.history,
    }


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for the Doxascope network.")
    parser.add_argument("policy_name", help="Name of the policy to sweep.")
    parser.add_argument("num_future_timesteps", type=int, help="Number of future steps to predict.")
    parser.add_argument("--num-past-timesteps", type=int, default=0, help="Number of past steps to predict.")
    parser.add_argument("--num-configs", type=int, default=30, help="Number of random configurations to test.")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs for each run.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping in each trial.")
    parser.add_argument(
        "--sweep-type", type=str, default="hyper", choices=["hyper", "arch"], help="Type of sweep: 'hyper' or 'arch'."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("train_dir/doxascope/raw_data"),
        help="Directory containing the raw doxascope data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("train_dir/doxascope/sweeps"),
        help="Directory to save the sweep results.",
    )
    parser.add_argument("--test-split", type=float, default=0.15, help="Proportion of data for the test set.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Proportion of data for the validation set.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda').")

    args = parser.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a unique directory for this sweep run
    sweep_name = f"sweep_{args.sweep_type}_{time.strftime('%Y%m%d-%H%M%S')}"
    sweep_output_dir = args.output_dir / args.policy_name / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving sweep results to: {sweep_output_dir}")

    # Prepare data once for the entire sweep
    policy_data_dir = args.raw_data_dir / args.policy_name
    data_loaders = prepare_data(
        policy_data_dir,
        sweep_output_dir,  # Use sweep dir for preprocessed data cache
        args.test_split,
        args.val_split,
        args.num_future_timesteps,
        args.num_past_timesteps,
    )
    if data_loaders[0] is None:
        print("❌ Failed to prepare data. Aborting sweep.")
        return

    search_space = get_search_space(args.sweep_type)
    all_results = []

    for i in range(args.num_configs):
        config = sample_config(search_space)
        result = run_sweep_trial(i, args.num_configs, config, args, device, data_loaders)
        all_results.append(result)

    # Save final results
    results_df = pd.DataFrame([res for res in all_results if res["success"]])
    if not results_df.empty:
        results_df = results_df.sort_values(by="best_val_acc", ascending=False)
        results_df.to_csv(sweep_output_dir / "sweep_summary.csv", index=False)
        print(f"\n✅ Sweep summary saved to {sweep_output_dir / 'sweep_summary.csv'}")

        print("\n🏆 Top 5 Configurations (by Validation Accuracy):")
        print(results_df.head(5)[["best_val_acc", "test_acc_avg", "config"]])
    else:
        print("\nNo successful trials to summarize.")

    # Save all raw results to JSON
    with open(sweep_output_dir / "all_trial_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: str(o))  # Handle non-serializable types


if __name__ == "__main__":
    main()
