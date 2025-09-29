#!/usr/bin/env python3
"""
Doxascope Command-Line Interface

A unified, interactive CLI for training, analyzing, and managing Doxascope models.
"""

import argparse
import json
import os
import random
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# It's better to import functions directly from the modules
# to make the CLI script the single source of truth for argument parsing.
from .doxascope_network import (
    DoxascopeNet,
    DoxascopeTrainer,
    create_baseline_data,
    prepare_data,
)


def _prompt_for_value(prompt: str, target_type: type, default: Any) -> Any:
    """Prompts the user for a value and casts it to the target type."""
    while True:
        try:
            user_input = input(f"  -> Enter {prompt} (default: {default}): ")
            if not user_input:
                return default

            if target_type == bool:
                return user_input.lower() in ["y", "yes", "true", "1"]
            return target_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a value of type {target_type.__name__}.")
        except (KeyboardInterrupt, EOFError):
            print("\nConfiguration cancelled.")
            return None


def _select_command(choices: Dict[str, argparse.ArgumentParser]) -> Optional[str]:
    """Displays a list of commands with descriptions and prompts the user to select one."""
    print("\nPlease select a command:")
    commands = list(choices.keys())
    for i, name in enumerate(commands):
        help_text = choices[name].description or ""
        print(f"  [{i + 1}] {name:<10} - {help_text}")

    while True:
        try:
            choice = input(f"\nEnter number (1-{len(commands)}): ")
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(commands):
                return commands[idx]
            else:
                print("Invalid number, please try again.")
        except (ValueError, IndexError):
            print("Invalid input, please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def _interactive_prediction_config(args: argparse.Namespace) -> Optional[argparse.Namespace]:
    """Asks the user to confirm or change prediction timestep settings."""
    # Set a sensible default if num_future_timesteps is not provided (e.g., in interactive sweep)
    if getattr(args, "num_future_timesteps", None) is None:
        args.num_future_timesteps = 1

    print("\n--- Prediction Timestep Configuration ---")
    print(f"  - Future Timesteps to Predict : {args.num_future_timesteps}")
    print(f"  - Past Timesteps to Predict   : {args.num_past_timesteps}")
    print("-" * 39)

    proceed = input("Proceed with these settings? (Y/n): ").lower()
    if proceed in ["", "y", "yes"]:
        return args

    print("Enter new values or press Enter to keep the default.")
    new_future = _prompt_for_value("Future Timesteps", int, args.num_future_timesteps)
    if new_future is None:
        return None  # User cancelled
    args.num_future_timesteps = new_future

    new_past = _prompt_for_value("Past Timesteps", int, args.num_past_timesteps)
    if new_past is None:
        return None  # User cancelled
    args.num_past_timesteps = new_past

    print("\nPrediction settings updated.")
    return args


def handle_collect_command(args: argparse.Namespace):
    """Handles the 'collect' command."""
    print("\n--- Collect Doxascope Data ---")
    policy_uri = input("  -> Enter the policy URI to evaluate: ")
    if not policy_uri:
        print("No policy URI provided. Aborting collection.")
        return

    command_str = (
        f"uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri={policy_uri} doxascope_enabled=true"
    )

    print("\nHanding off to evaluation tool...")
    print(f"  Command: {command_str}\n")

    # Use os.execvp to replace the current process with the new command.
    # This will stream the output directly to the user's terminal.
    # shlex.split is used to handle quoted arguments correctly.
    try:
        args = shlex.split(command_str)
        os.execvp(args[0], args)
    except FileNotFoundError:
        print(f"\nError: Command '{args[0]}' not found.")
        print("Please ensure 'uv' is installed and you are running from the repository root.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to execute the command: {e}")


def _interactive_train_config(args: argparse.Namespace) -> Optional[argparse.Namespace]:
    """Shows current training settings and allows the user to change them interactively."""
    # Define the settings that can be changed, grouped for readability.
    config_groups = {
        "Data Splits": [
            ("test_split", "Test Split", float),
            ("val_split", "Validation Split", float),
        ],
        "Training Hyperparameters": [
            ("batch_size", "Batch Size", int),
            ("learning_rate", "Learning Rate", float),
            ("num_epochs", "Max Epochs", int),
            ("patience", "Early Stopping Patience", int),
        ],
        "Model Architecture": [
            ("hidden_dim", "Hidden Dimension", int),
            ("dropout_rate", "Dropout Rate", float),
            ("activation_fn", "Activation Function (silu, relu, gelu)", str),
            ("main_net_depth", "Main Network Depth", int),
            ("processor_depth", "Processor Network Depth", int),
        ],
        "Other Settings": [
            ("run_name", "Run Name (leave blank for auto)", str),
            ("train_random_baseline", "Train Random Baseline (y/n)", bool),
            ("force_reprocess", "Force Data Reprocessing (y/n)", bool),
        ],
    }

    # First, display the current settings
    print("\n--- Other Training Configuration ---")
    for group, options in config_groups.items():
        print(f"\n  {group}:")
        for key, prompt, _ in options:
            print(f"    - {prompt:<28}: {getattr(args, key)}")
    print("-" * 36)

    # Ask if the user wants to proceed
    proceed = input("Proceed with these settings? (Y/n): ").lower()
    if proceed in ["", "y", "yes"]:
        return args

    # Enter interactive editing mode
    flat_options = [opt for group in config_groups.values() for opt in group]
    while True:
        print("\n--- Edit Configuration ---")
        for i, (key, prompt, _) in enumerate(flat_options):
            print(f"  [{i + 1}] {prompt:<28}: {getattr(args, key)}")
        print("\n  [D] Done, start training")
        print("  [Q] Quit")

        choice = input("Select an option to change: ").lower()
        if choice in ["d", "done"]:
            return args
        if choice in ["q", "quit"]:
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(flat_options):
                key, prompt, target_type = flat_options[idx]
                current_value = getattr(args, key)
                new_value = _prompt_for_value(prompt, target_type, current_value)
                if new_value is not None:
                    setattr(args, key, new_value)
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input. Please enter a number, 'D', or 'Q'.")


def _select_string_from_list(items: List[str], item_type: str) -> Optional[str]:
    """Displays a list of strings and prompts the user to select one."""
    if not items:
        print(f"No {item_type}s found.")
        return None

    print(f"Please select a {item_type}:")
    for i, item in enumerate(items):
        print(f"  [{i + 1}] {item}")

    while True:
        try:
            choice = input(f"Enter number (1-{len(items)}): ")
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            else:
                print("Invalid number, please try again.")
        except (ValueError, IndexError):
            print("Invalid input, please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


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


def get_available_policies(data_dir: Path) -> List[Path]:
    """Scans a directory and returns a list of policy subdirectories."""
    if not data_dir.is_dir():
        return []
    return sorted([d for d in data_dir.iterdir() if d.is_dir()])


def get_available_runs(policy_dir: Path) -> List[Path]:
    """Scans a policy directory and returns a list of run subdirectories."""
    if not policy_dir.is_dir():
        return []
    return sorted([d for d in policy_dir.iterdir() if d.is_dir() and (d / "best_model.pth").exists()])


def select_from_list(items: List[Path], item_type: str) -> Optional[Path]:
    """Displays a list of items and prompts the user to select one."""
    if not items:
        print(f"No {item_type}s found.")
        return None

    print(f"Please select a {item_type}:")
    for i, item in enumerate(items):
        print(f"  [{i + 1}] {item.name}")

    while True:
        try:
            choice = input(f"Enter number (1-{len(items)}): ")
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            else:
                print("Invalid number, please try again.")
        except (ValueError, IndexError):
            print("Invalid input, please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def find_latest_run(policy_dir: Path) -> Optional[Path]:
    """Finds the most recent run directory in a policy's result folder."""
    run_dirs = get_available_runs(policy_dir)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.name)


# --- Analysis Functions (from doxascope_analysis.py) ---


def load_data_and_model(
    policy_dir: Path,
    device: str,
    model_filename: str = "best_model.pth",
    results_filename: str = "test_results.json",
    history_filename: str = "training_history.csv",
):
    """Loads the model, data, and results for a given policy."""
    model_path = policy_dir / model_filename
    test_data_path = policy_dir / "preprocessed_data" / "test.npz"
    history_path = policy_dir / history_filename
    test_results_path = policy_dir / results_filename

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = DoxascopeNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Load test data
    test_data = np.load(test_data_path)
    X_test, y_test = test_data["X"], test_data["y"]

    # Load history and test results
    history = pd.read_csv(history_path).to_dict(orient="list")
    with open(test_results_path, "r") as f:
        test_results = json.load(f)

    return model, X_test, y_test, history, test_results


def get_predictions(model: DoxascopeNet, X_test: np.ndarray, device: str):
    """Generates predictions from the model."""
    X_test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)

    predicted_indices = [torch.argmax(o, dim=1).cpu().numpy() for o in outputs]
    probabilities = [torch.softmax(o, dim=1).cpu().numpy() for o in outputs]

    return predicted_indices, probabilities


def plot_training_history(history: dict, output_path: Path):
    """Plots and saves the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot Accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    fig.suptitle("Training History")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path)
    plt.close()


def plot_multistep_accuracy(test_results: dict, output_path: Path, baseline_results: Optional[dict] = None):
    """Plots the test accuracy for each predicted timestep."""
    timesteps = test_results["timesteps"]
    accuracies = test_results["test_accuracy_per_step"]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, accuracies, marker="o", linestyle="-", label="Doxascope")

    if baseline_results:
        baseline_accuracies = baseline_results["test_accuracy_per_step"]
        plt.plot(timesteps, baseline_accuracies, marker="x", linestyle="--", label="Random Baseline")

    plt.title("Test Accuracy per Timestep")
    plt.xlabel("Timestep Relative to Present")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Ensure all timestep labels are shown
    if timesteps:
        plt.xticks(timesteps)

    plt.savefig(output_path)
    plt.close()


def generate_all_plots(policy_dir: Path, device: str):
    """
    Loads all necessary data for a single run and generates all standard analysis plots.
    """
    analysis_dir = policy_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating analysis plots in: {analysis_dir}")

    try:
        model, X_test, y_test, history, test_results = load_data_and_model(policy_dir, device)
        predicted_indices, _ = get_predictions(model, X_test, device)
    except FileNotFoundError as e:
        print(f"Skipping plot generation for {policy_dir.name}: {e}")
        return

    # Generate standard plots
    plot_training_history(history, analysis_dir / "training_history.png")
    plot_multistep_accuracy(test_results, analysis_dir / "multistep_accuracy.png")

    # Handle baseline comparison if data exists
    baseline_results_path = policy_dir / "test_results_baseline.json"
    if baseline_results_path.exists():
        with open(baseline_results_path, "r") as f:
            baseline_results = json.load(f)
        plot_multistep_accuracy(
            test_results,
            analysis_dir / "multistep_accuracy_comparison.png",
            baseline_results=baseline_results,
        )
    print(f"Successfully generated plots for run {policy_dir.name}")


def compare_policies(policy_names: List[str], data_dir: Path, output_dir: Path):
    """
    Compares the latest runs of multiple policies against each other.
    """
    latest_run_results = {}
    for name in policy_names:
        policy_dir = data_dir / name
        latest_run = find_latest_run(policy_dir)
        if latest_run is None:
            print(f"Warning: No runs found for policy '{name}'. Skipping.")
            continue
        results_path = latest_run / "test_results.json"
        if results_path.exists():
            print(f"Using latest run for {name}: {latest_run.name}")
            with open(results_path, "r") as f:
                latest_run_results[name] = json.load(f)
        else:
            print(f"Warning: 'test_results.json' not found in latest run for '{name}'. Skipping.")

    if not latest_run_results:
        print("No valid runs found to compare.")
        return

    plt.figure(figsize=(12, 8))
    for policy_name, results in latest_run_results.items():
        if "timesteps" in results and "test_accuracy_per_step" in results:
            plt.plot(
                results["timesteps"],
                results["test_accuracy_per_step"],
                marker="o",
                linestyle="-",
                label=f"Policy: {policy_name}",
            )

    plt.title("Latest Run Accuracy Comparison")
    plt.xlabel("Timestep Relative to Present")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Create a filename-safe name for the plot
    safe_name = "_vs_".join(p.replace("/", "_") for p in policy_names)
    comparison_plot_path = output_dir / f"comparison_{safe_name}.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Policy comparison plot saved to {comparison_plot_path}")


def compare_runs(policy_dir: Path, output_dir: Path):
    """Compares the multistep accuracy of all runs for a single policy."""
    run_results = {}
    for run_dir in sorted(policy_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "test_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                run_results[run_dir.name] = json.load(f)

    if not run_results:
        print(f"No completed runs found in {policy_dir}")
        return

    plt.figure(figsize=(12, 8))
    for run_name, results in sorted(run_results.items()):
        if "timesteps" in results and "test_accuracy_per_step" in results:
            plt.plot(
                results["timesteps"],
                results["test_accuracy_per_step"],
                marker="o",
                linestyle="-",
                label=f"Run: {run_name}",
            )

    plt.title(f"Multistep Accuracy Comparison for Policy: {policy_dir.name}")
    plt.xlabel("Timestep Relative to Present")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    comparison_plot_path = output_dir / f"comparison_{policy_dir.name}.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Comparison plot saved to {comparison_plot_path}")


# --- Command Handlers ---


def handle_train_command(args: argparse.Namespace):
    """Handles the 'train' command."""
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    policy_name = args.policy_name
    # If no policy name is provided, enter interactive selection mode
    if not policy_name:
        available_policies = get_available_policies(args.raw_data_dir)
        selected_policy_path = select_from_list(available_policies, "policy to train")
        if not selected_policy_path:
            print("No policy selected. Aborting training.")
            return
        policy_name = selected_policy_path.name
        # After interactive policy selection, allow interactive config editing
        updated_args = _interactive_prediction_config(args)
        if updated_args is None:
            print("Configuration cancelled. Aborting training.")
            return
        args = updated_args

        updated_args = _interactive_train_config(args)
        if updated_args is None:
            print("Configuration cancelled. Aborting training.")
            return
        args = updated_args

    # Prepare data and output directories
    policy_data_dir = args.raw_data_dir / policy_name
    if not policy_data_dir.is_dir():
        print(f"Error: Raw data directory for policy '{policy_name}' not found at {policy_data_dir}")
        print(
            "Please ensure you have collected data for this policy by running an evaluation with 'doxascope_enabled=true'."
        )
        return

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / policy_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # Prepare data loaders for the main model
    print("\n--- Preparing Data for Main Model ---")
    data_loaders = prepare_data(
        raw_data_dir=policy_data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        test_split=args.test_split,
        val_split=args.val_split,
        num_future_timesteps=args.num_future_timesteps,
        num_past_timesteps=args.num_past_timesteps,
        data_split_seed=42,
        force_reprocess=args.force_reprocess,
    )

    if data_loaders[0] is None:
        print("Failed to create data loaders. Aborting.")
        return

    # Run the main training pipeline
    run_training_pipeline(
        policy_name=policy_name,
        output_dir=output_dir,
        device=device,
        args=args,
        data_loaders=data_loaders,
        is_baseline=False,
    )

    # Run the baseline training pipeline if requested
    if args.train_random_baseline:
        print("\n--- Preparing Data for Baseline Model (using randomized inputs) ---")
        # Create baseline data by randomizing the preprocessed inputs
        preprocessed_dir = output_dir / "preprocessed_data"
        baseline_data_loaders = create_baseline_data(preprocessed_dir, args.batch_size)

        if baseline_data_loaders[0] is None:
            print("Failed to create data loaders for the baseline model. Aborting baseline training.")
            return

        run_training_pipeline(
            policy_name=policy_name,
            output_dir=output_dir,
            device=device,
            args=args,
            data_loaders=baseline_data_loaders,
            is_baseline=True,
        )


def handle_analyze_command(args: argparse.Namespace):
    """Handles the 'analyze' command."""
    policy_dir = args.data_dir / args.policy_name if args.policy_name else None

    # --- Interactive Selection ---
    if not policy_dir:
        policies = get_available_policies(args.data_dir)
        selected_policy = select_from_list(policies, "policy")
        if not selected_policy:
            return
        policy_dir = selected_policy

    if not policy_dir.is_dir():
        print(f"Error: Policy directory not found at {policy_dir}")
        return

    if args.run_name:
        policy_run_dir = policy_dir / args.run_name
        if not policy_run_dir.is_dir():
            print(f"Error: Run directory not found at {policy_run_dir}")
            return
    else:
        print(f"No run name specified, finding the latest run for policy '{policy_dir.name}'...")
        runs = get_available_runs(policy_dir)
        policy_run_dir = select_from_list(runs, f"run for policy '{policy_dir.name}'")
        if not policy_run_dir:
            return

    print(f"Analyzing run: {policy_run_dir.name}")
    generate_all_plots(policy_run_dir, args.device)


def handle_compare_command(args: argparse.Namespace):
    """Handles the 'compare' command."""
    if len(args.policy_names) == 1:
        # Compare all runs for a single policy
        policy_name = args.policy_names[0]
        policy_dir = args.data_dir / policy_name
        compare_runs(policy_dir, policy_dir)
    else:
        # Compare the latest run of multiple policies
        compare_policies(args.policy_names, args.data_dir, args.data_dir)


def handle_sweep_command(args: argparse.Namespace):
    """Handles the 'sweep' command."""
    policy_name = args.policy_name

    # --- Interactive Mode ---
    if not policy_name:
        # 1. Select Policy
        available_policies = get_available_policies(args.raw_data_dir)
        selected_policy_path = select_from_list(available_policies, "policy to sweep")
        if not selected_policy_path:
            print("No policy selected. Aborting sweep.")
            return
        policy_name = selected_policy_path.name

        # 2. Configure prediction timesteps
        updated_args = _interactive_prediction_config(args)
        if updated_args is None:
            print("Configuration cancelled. Aborting sweep.")
            return
        args = updated_args

        # 3. Select sweep type
        sweep_type = _select_string_from_list(["hyper", "arch"], "sweep type")
        if sweep_type is None:
            print("No sweep type selected. Aborting sweep.")
            return
        args.sweep_type = sweep_type

        # 4. Configure sweep parameters
        updated_args = _interactive_sweep_config(args)
        if updated_args is None:
            print("Configuration cancelled. Aborting sweep.")
            return
        args = updated_args

    # --- Non-interactive mode ---
    elif args.num_future_timesteps is None:
        print("Error: 'num_future_timesteps' is required when running sweep non-interactively.")
        print("Usage: uv run doxascope sweep <policy_name> <num_future_timesteps>")
        return

    # Create a unique directory for this sweep run
    sweep_name = f"sweep_{args.sweep_type}_{time.strftime('%Y%m%d-%H%M%S')}"
    sweep_output_dir = args.output_dir / policy_name / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving sweep results to: {sweep_output_dir}")

    # Prepare data once for the entire sweep
    policy_data_dir = args.raw_data_dir / policy_name
    data_loaders = prepare_data(
        policy_data_dir,
        sweep_output_dir,  # Use sweep dir for preprocessed data cache
        args.batch_size,
        args.test_split,
        args.val_split,
        args.num_future_timesteps,
        args.num_past_timesteps,
        force_reprocess=args.force_reprocess,
    )
    if data_loaders[0] is None:
        print("❌ Failed to prepare data. Aborting sweep.")
        return

    search_space = get_search_space(args.sweep_type)
    all_results = []

    for i in range(args.num_configs):
        config = sample_config(search_space)
        result = run_sweep_trial(i, args.num_configs, config, args, args.device, data_loaders)
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


def main():
    parser = argparse.ArgumentParser(
        description="Doxascope: A tool for investigating the internal states of RL agents.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Collect Command ---
    parser_collect = subparsers.add_parser(
        "collect",
        help="Collect training data by running a policy evaluation.",
        description="Collect training data by running a policy evaluation.",
    )
    parser_collect.set_defaults(func=handle_collect_command)

    # --- Train Command ---
    parser_train = subparsers.add_parser(
        "train",
        help="Train a Doxascope network on collected data.",
        description="Train a Doxascope network on collected data.",
    )
    parser_train.add_argument(
        "policy_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the policy to train on (interactive if omitted).",
    )
    parser_train.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("train_dir/doxascope/raw_data"),
        help="Directory containing the raw doxascope data files.",
    )
    parser_train.add_argument(
        "--output-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory to save the preprocessed data and model checkpoints.",
    )
    parser_train.add_argument("--run-name", type=str, default=None, help="Unique name for this training run.")
    parser_train.add_argument("--test-split", type=float, default=0.15, help="Proportion of data for the test set.")
    parser_train.add_argument(
        "--val-split", type=float, default=0.15, help="Proportion of data for the validation set."
    )
    parser_train.add_argument(
        "--num-future-timesteps", type=int, default=1, help="Number of future timesteps to predict."
    )
    parser_train.add_argument("--num-past-timesteps", type=int, default=0, help="Number of past timesteps to predict.")
    parser_train.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser_train.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser_train.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for.")
    parser_train.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser_train.add_argument(
        "--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda')"
    )
    parser_train.add_argument(
        "--train-random-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train a baseline model with random memory vectors for comparison.",
    )
    parser_train.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for the model.")
    parser_train.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout rate for the model.")
    parser_train.add_argument("--activation_fn", type=str, default="silu", help="Activation function for the model.")
    parser_train.add_argument("--main_net_depth", type=int, default=3, help="Depth of the main network.")
    parser_train.add_argument("--processor_depth", type=int, default=1, help="Depth of the state processors.")
    parser_train.add_argument(
        "--force-reprocess", action="store_true", help="Force reprocessing of raw data even if cache exists."
    )
    parser_train.set_defaults(func=handle_train_command)

    # --- Analyze Command (placeholder) ---
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a trained network's performance and generate plots.",
        description="Analyze a trained network's performance and generate plots.",
    )
    parser_analyze.add_argument(
        "policy_name", type=str, nargs="?", default=None, help="Name of the policy to analyze (optional)."
    )
    parser_analyze.add_argument(
        "run_name", type=str, nargs="?", default=None, help="Name of the run to analyze (optional)."
    )
    parser_analyze.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing policy results.",
    )
    parser_analyze.add_argument(
        "--device", type=str, default="auto", help="Device to use for analysis (e.g., 'cpu', 'cuda')."
    )
    parser_analyze.set_defaults(func=handle_analyze_command)

    # --- Compare Command ---
    parser_compare = subparsers.add_parser(
        "compare",
        help="Compare the performance of different policies or training runs.",
        description="Compare the performance of different policies or training runs.",
    )
    parser_compare.add_argument("policy_names", nargs="+", help="Name(s) of the policy/policies to compare.")
    parser_compare.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing policy results.",
    )
    parser_compare.set_defaults(func=handle_compare_command)

    # --- Sweep Command ---
    parser_sweep = subparsers.add_parser(
        "sweep",
        help="Run a hyperparameter or architecture sweep to find the best model.",
        description="Run a hyperparameter or architecture sweep to find the best model.",
    )
    parser_sweep.add_argument(
        "policy_name", nargs="?", default=None, type=str, help="Name of the policy to sweep (interactive if omitted)."
    )
    parser_sweep.add_argument(
        "num_future_timesteps",
        type=int,
        nargs="?",
        default=None,
        help="Number of future steps to predict (interactive if omitted).",
    )
    parser_sweep.add_argument("--num-past-timesteps", type=int, default=0, help="Number of past steps to predict.")
    parser_sweep.add_argument("--num-configs", type=int, default=30, help="Number of random configurations to test.")
    parser_sweep.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs for each run.")
    parser_sweep.add_argument("--patience", type=int, default=10, help="Patience for early stopping in each trial.")
    parser_sweep.add_argument(
        "--sweep-type", type=str, default="hyper", choices=["hyper", "arch"], help="Type of sweep: 'hyper' or 'arch'."
    )
    parser_sweep.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("train_dir/doxascope/raw_data"),
        help="Directory containing raw data files.",
    )
    parser_sweep.add_argument(
        "--output-dir", type=Path, default=Path("train_dir/doxascope/sweeps"), help="Directory to save sweep results."
    )
    parser_sweep.add_argument("--test-split", type=float, default=0.15, help="Proportion of data for the test set.")
    parser_sweep.add_argument(
        "--val-split", type=float, default=0.15, help="Proportion of data for the validation set."
    )
    parser_sweep.add_argument(
        "--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda')."
    )
    parser_sweep.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser_sweep.add_argument(
        "--force-reprocess", action="store_true", help="Force reprocessing of raw data even if cache exists."
    )
    parser_sweep.set_defaults(func=handle_sweep_command)

    args = parser.parse_args()
    # Auto-detect device if not specified
    if hasattr(args, "device") and args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # If no command is given, prompt the user to select one
    if args.command is None:
        command_name = _select_command(subparsers.choices)
        if not command_name:
            print("No command selected. Exiting.")
            return
        # Re-parse args for the chosen command to populate its defaults.
        # This will fail if the command has required args not provided,
        # which is why we made policy_name optional.
        args = parser.parse_args([command_name])

    args.func(args)


if __name__ == "__main__":
    main()
