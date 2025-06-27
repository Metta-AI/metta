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

import numpy as np
import torch
from torch.utils.data import DataLoader

from .doxascope_network import DoxascopeDataset, DoxascopeNet, DoxascopeTrainer


class DoxascopeSweep:
    """Manages the hyperparameter sweep experiment."""

    def __init__(self, policy_name: str, num_future_timesteps: int, sweep_type: str = "hyper"):
        self.policy_name = policy_name
        self.num_future_timesteps = num_future_timesteps
        self.results = []
        self.sweep_type = sweep_type
        self.output_dir = Path(f"doxascope/data/results/{policy_name}")

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_data()

    def load_data(self):
        """Load pre-split train, validation, and test data."""
        print("📊 Loading training, validation, and test data...")
        preprocessed_dir = self.output_dir / "preprocessed_data"
        try:
            train_data = np.load(preprocessed_dir / "train_data.npz")
            val_data = np.load(preprocessed_dir / "val_data.npz")
            test_data = np.load(preprocessed_dir / "test_data.npz")
        except FileNotFoundError as e:
            print(f"❌ Error: Data file not found: {e}. Policy '{self.policy_name}' may not have preprocessed data.")
            raise SystemExit

        print(
            f"Datasets loaded: {len(train_data['X'])} train, {len(val_data['X'])} val, {len(test_data['X'])} test samples."
        )
        return (
            train_data["X"],
            train_data["y"],
            val_data["X"],
            val_data["y"],
            test_data["X"],
            test_data["y"],
        )

    def _define_hyperparameter_search_space(self):
        """Define the hyperparameter search space for random search."""
        return {
            "hidden_dim": [128, 256, 384, 512],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "lr": [0.0001, 0.0005, 0.001, 0.005],
            "batch_size": [32, 64, 128],
            "weight_decay": [1e-6, 1e-5, 1e-4],
        }

    def _define_architectural_search_space(self):
        """Define the architectural search space."""
        # Fix hyperparameters to the best found, and focus on architecture
        hyperparams = {
            "hidden_dim": [512],
            "dropout_rate": [0.4],
            "lr": [0.0007],
            "batch_size": [32],
        }

        architectures = {
            "activation_fn": ["gelu", "relu", "silu"],
            "main_net_depth": [1, 2, 3],
            "processor_depth": [1, 2],
        }

        search_space = {**hyperparams, **architectures}
        return search_space

    def define_search_space(self):
        """Define the hyperparameter search space based on the sweep type."""
        if self.sweep_type == "arch":
            print("🔬 Using architectural search space.")
            return self._define_architectural_search_space()

        print("⚙️ Using hyperparameter search space.")
        return self._define_hyperparameter_search_space()

    def sample_configurations(self, num_samples=50):
        """Sample random configurations from the search space."""
        search_space = self.define_search_space()
        configs = []
        for _ in range(num_samples):
            config = {param: random.choice(values) for param, values in search_space.items()}
            configs.append(config)
        return configs

    def run_single_config(self, config: dict, max_epochs=50):
        """Train a single configuration and return the results."""
        try:
            # Create datasets
            train_dataset = DoxascopeDataset(self.X_train, self.y_train)
            val_dataset = DoxascopeDataset(self.X_val, self.y_val)
            test_dataset = DoxascopeDataset(self.X_test, self.y_test)

            # Create data loaders
            batch_size = config.get("batch_size", 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Build a dictionary of network parameters that are present in the sweep's config.
            # This allows DoxascopeNet to use its own defaults for any params not in the sweep.
            net_params = [
                "hidden_dim",
                "dropout_rate",
                "activation_fn",
                "main_net_depth",
                "processor_depth",
            ]
            model_config = {key: config[key] for key in net_params if key in config}

            # Create model
            model = DoxascopeNet(
                input_dim=self.X_train.shape[1],
                num_future_timesteps=self.num_future_timesteps,
                **model_config,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = DoxascopeTrainer(model, device=device)

            # Train with early stopping
            start_time = time.time()
            history, _, _ = trainer.train(train_loader, val_loader, num_epochs=max_epochs, lr=config.get("lr", 0.0007))
            train_time = time.time() - start_time

            best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0

            # Evaluate on the test set
            _, test_acc_per_step, _, _ = trainer._run_epoch(test_loader, is_training=False)
            test_acc_avg = sum(test_acc_per_step) / len(test_acc_per_step) if test_acc_per_step else 0

            return {
                "config": config,
                "best_val_acc": best_val_acc,
                "test_acc_avg": test_acc_avg,
                "test_acc_per_step": test_acc_per_step,
                "train_time": train_time,
                "success": True,
            }

        except Exception as e:
            # Log the error for debugging
            print(f"   ❌ Failed with error: {e}")
            return {"config": config, "error": str(e), "success": False}

    def run_sweep(self, num_configs=30, max_epochs=50):
        """Run the hyperparameter sweep."""
        print(f"🚀 Starting hyperparameter sweep for policy '{self.policy_name}'...")
        configs = self.sample_configurations(num_configs)
        print(f"📊 Testing {len(configs)} configurations.")

        for i, config in enumerate(configs):
            print(f"\n⚡ [{i + 1}/{len(configs)}] Testing Config: {config}")
            result = self.run_single_config(config, max_epochs=max_epochs)
            if result["success"]:
                print(f"   ✅ Val Acc: {result['best_val_acc']:.2f}%, Test Acc (avg): {result['test_acc_avg']:.2f}%")
                if result.get("test_acc_per_step"):
                    print(f"      t+1: {result['test_acc_per_step'][0]:.2f}%")
                    if len(result["test_acc_per_step"]) > 1:
                        k = len(result["test_acc_per_step"])
                        print(f"      t+{k}: {result['test_acc_per_step'][-1]:.2f}%")
            self.results.append(result)

        self.save_results()

    def save_results(self):
        """Save sweep results to a JSON file."""
        if self.sweep_type == "arch":
            filename = "doxascope_sweep_results_architectural.json"
        else:
            filename = "doxascope_sweep_results.json"

        output_file = self.output_dir / filename

        # Make results JSON-serializable
        serializable_results = []
        for result in self.results:
            if result["success"]:
                # Create a copy to modify
                res_copy = result.copy()
                # Convert numpy types if they exist
                for k, v in res_copy.items():
                    if isinstance(v, np.floating):
                        res_copy[k] = float(v)
                serializable_results.append(res_copy)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Sweep results saved to {output_file}")

        # Print best result
        successful_results = [r for r in self.results if r.get("success")]
        if successful_results:
            best_run = max(successful_results, key=lambda r: r["best_val_acc"])
            print("\n🏆 Best Configuration (by Validation Accuracy):")
            print(f"   Config: {best_run['config']}")
            print(f"   Validation Accuracy: {best_run['best_val_acc']:.2f}%")
            print(f"   Test Accuracy (avg): {best_run['test_acc_avg']:.2f}%")
            if best_run.get("test_acc_per_step"):
                print(f"   Test Accuracy (t+1): {best_run['test_acc_per_step'][0]:.2f}%")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for the Doxascope network.")
    parser.add_argument("policy_name", help="Name of the policy to sweep.")
    parser.add_argument("num_future_timesteps", type=int, help="Number of future steps to predict.")
    parser.add_argument("--num-configs", type=int, default=30, help="Number of random configurations to test.")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs for each run.")
    parser.add_argument(
        "--sweep-type",
        type=str,
        default="hyper",
        choices=["hyper", "arch"],
        help="Type of sweep: 'hyper' or 'arch'.",
    )
    args = parser.parse_args()

    sweeper = DoxascopeSweep(
        policy_name=args.policy_name,
        num_future_timesteps=args.num_future_timesteps,
        sweep_type=args.sweep_type,
    )
    sweeper.run_sweep(num_configs=args.num_configs, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
