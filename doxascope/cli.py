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

import pandas as pd
import torch

from .doxascope_analysis import compare_policies, compare_runs
from .doxascope_network import DoxascopeNet, DoxascopeTrainer, create_baseline_data, prepare_data
from .doxascope_plots import generate_all_plots


def _prompt_for_value(prompt: str, target_type: type, default: Any) -> Any:
    """Prompts the user for a value and casts it to the target type."""
    while True:
        try:
            user_input = input(f"  -> Enter {prompt} (default: {default}): ")
            if not user_input:
                return default

            if target_type is bool:
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

    proceed = input("Proceed with these settings? (y/n): ").lower()
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

    while True:
        try:
            user_input = input("  -> Enter number of simulations to run (default: 10): ")
            if not user_input:
                num_simulations = 10
                break
            num_simulations = int(user_input)
            if num_simulations < 1:
                print("  !! Number of simulations must be at least 1.")
                continue
            if num_simulations < 5:
                print("  -> Warning: For decent training, at least 5 simulations are recommended.")
            break
        except ValueError:
            print("  !! Invalid input. Please enter an integer.")
        except (KeyboardInterrupt, EOFError):
            print("\nCollection cancelled.")
            return

    command_str = f"uv run ./tools/run.py doxascope.evaluate policy_uri={policy_uri} num_simulations={num_simulations}"

    print("\nHanding off to evaluation tool...")
    print(f"  Command: {command_str}\n")

    try:
        argv = shlex.split(command_str)
        os.execvp(argv[0], argv)  # type: ignore[arg-type]
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
    proceed = input("Proceed with these settings? (y/n): ").lower()
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


def _select_multiple_from_list(items: List[Path], item_type: str) -> Optional[List[Path]]:
    """Displays a list of items and prompts the user to select one or more."""
    if not items:
        print(f"No {item_type}s found.")
        return None

    print(f"Please select {item_type}s to compare (e.g., '1 3 4', 'all'):")
    for i, item in enumerate(items):
        print(f"  [{i + 1}] {item.name}")

    while True:
        try:
            choice_str = input(f"Enter numbers or 'all' (1-{len(items)}): ")
            if not choice_str:
                return None

            if choice_str.lower() == "all":
                return items

            indices = [int(i) - 1 for i in choice_str.split()]
            if all(0 <= idx < len(items) for idx in indices):
                return [items[idx] for idx in indices]
            else:
                print("Invalid number(s), please try again.")
        except (ValueError, IndexError):
            print("Invalid input, please enter space-separated numbers or 'all'.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def _interactive_sweep_config(args: argparse.Namespace) -> Optional[argparse.Namespace]:
    """Shows current sweep settings and allows the user to change them interactively."""
    print("\n--- Sweep Configuration ---")
    print(f"  - Number of configurations to test: {args.num_configs}")
    print(f"  - Max epochs per trial: {args.max_epochs}")
    print(f"  - Early stopping patience per trial: {args.patience}")
    print("-" * 39)

    proceed = input("Proceed with these settings? (y/n): ").lower()
    if proceed in ["", "y", "yes"]:
        return args

    print("Enter new values or press Enter to keep the default.")
    new_num_configs = _prompt_for_value("Number of configs", int, args.num_configs)
    if new_num_configs is None:
        return None
    args.num_configs = new_num_configs

    new_max_epochs = _prompt_for_value("Max epochs", int, args.max_epochs)
    if new_max_epochs is None:
        return None
    args.max_epochs = new_max_epochs

    new_patience = _prompt_for_value("Patience", int, args.patience)
    if new_patience is None:
        return None
    args.patience = new_patience

    print("\nSweep settings updated.")
    return args


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
        "activation_fn": ["gelu"],
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

    # Separate model params from training params (like learning rate)
    model_config = config.copy()
    trial_lr = model_config.pop("lr", 0.001)

    model_params = {
        "input_dim": input_dim,
        "num_future_timesteps": args.num_future_timesteps,
        "num_past_timesteps": args.num_past_timesteps,
        **model_config,
    }
    model = DoxascopeNet(**model_params).to(device)
    trainer = DoxascopeTrainer(model, device=device)

    start_time = time.time()
    training_result = trainer.train(
        train_loader, val_loader, num_epochs=args.max_epochs, lr=trial_lr, patience=args.patience
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


"""
# Analysis functions moved to doxascope_analysis.py
"""


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
        print("Please ensure data exists or collect it via 'doxascope collect' (doxascope_enabled=true).")
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
    )

    if data_loaders[0] is None:
        print("Failed to create data loaders. Aborting.")
        return

    # Run the main training pipeline
    train_loader, val_loader, test_loader, input_dim = data_loaders
    if input_dim is None:
        print("Error: input_dim is None. Cannot determine model input size.")
        return

    model_params = {
        "input_dim": input_dim,
        "num_future_timesteps": args.num_future_timesteps,
        "num_past_timesteps": args.num_past_timesteps,
        "hidden_dim": args.hidden_dim,
        "dropout_rate": args.dropout_rate,
        "activation_fn": args.activation_fn,
        "main_net_depth": args.main_net_depth,
        "processor_depth": args.processor_depth,
    }

    model = DoxascopeNet(**model_params).to(device)
    trainer = DoxascopeTrainer(model, device=device)
    print("\n--- Starting Training (Main) ---")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    main_run_artifacts = trainer.train_and_evaluate(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        patience=args.patience,
        output_dir=output_dir,
        policy_name=policy_name,
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

        train_loader_base, val_loader_base, test_loader_base, input_dim_base = baseline_data_loaders
        if input_dim_base is None:
            print("Error: input_dim is None for baseline. Cannot determine model input size.")
            return

        model_base = DoxascopeNet(**model_params).to(device)
        trainer_base = DoxascopeTrainer(model_base, device=device)
        print("\n--- Starting Training (Baseline) ---")
        print(f"Model Parameters: {sum(p.numel() for p in model_base.parameters() if p.requires_grad):,}")

        trainer_base.train_and_evaluate(
            train_loader=train_loader_base,
            val_loader=val_loader_base,
            test_loader=test_loader_base,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            patience=args.patience,
            output_dir=output_dir,
            policy_name=policy_name,
            is_baseline=True,
        )

    # --- Generate Analysis Plots ---
    # This is now done after the baseline run to ensure baseline results are available.
    if main_run_artifacts and main_run_artifacts["test_loader"]:
        print("\n--- Generating Analysis Plots ---")
        generate_all_plots(
            output_dir=output_dir,
            device=device,
            model=main_run_artifacts["model"],
            history=main_run_artifacts["history"],
            test_results=main_run_artifacts["test_results"],
            test_loader=main_run_artifacts["test_loader"],
            is_baseline=False,  # Always generate plots for the main model
        )


def handle_compare_command(args: argparse.Namespace):
    """Handles the 'compare' command."""
    # If no policies are specified, enter fully interactive mode
    if not args.policy_names:
        print("\nPlease select a comparison type:")
        choices = [
            "Compare runs within a single policy",
            "Compare latest runs across multiple policies",
        ]
        choice = _select_string_from_list(choices, "comparison type")

        if not choice:
            return

        if "single policy" in choice:
            available_policies = get_available_policies(args.data_dir)
            if not available_policies:
                print("No policies with completed runs found.")
                return
            policy_path = select_from_list(available_policies, "policy to compare")
            if not policy_path:
                return
            args.policy_names = [policy_path.name]
            # Fall through to the single-policy logic below

        elif "multiple policies" in choice:
            available_policies = get_available_policies(args.data_dir)
            if not available_policies:
                print("No policies with completed runs found.")
                return
            selected_policies = _select_multiple_from_list(available_policies, "policy")
            if not selected_policies or len(selected_policies) < 2:
                print("Please select at least two policies to compare.")
                return
            args.policy_names = [p.name for p in selected_policies]
            # Fall through to the multi-policy logic below
        else:
            return  # Should not happen

    if len(args.policy_names) == 1:
        # Compare selected runs for a single policy
        policy_name = args.policy_names[0]
        policy_dir = args.data_dir / policy_name
        available_runs = get_available_runs(policy_dir)

        selected_runs = _select_multiple_from_list(available_runs, "run")
        if not selected_runs:
            print("No runs selected for comparison.")
            return

        compare_runs(selected_runs, policy_name, policy_dir)
    else:
        # Compare the latest run of multiple policies
        compare_policies(args.policy_names, args.data_dir, args.data_dir)


def handle_sweep_command(args: argparse.Namespace):
    """Handles the 'sweep' command."""
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

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
    )
    if data_loaders[0] is None:
        print("❌ Failed to prepare data. Aborting sweep.")
        return

    # Unpack data loaders
    train_loader, val_loader, test_loader, input_dim = data_loaders
    data_loaders_for_sweep = (train_loader, val_loader, test_loader, input_dim)

    search_space = get_search_space(args.sweep_type)
    all_results = []

    for i in range(args.num_configs):
        config = sample_config(search_space)
        result = run_sweep_trial(i, args.num_configs, config, args, device, data_loaders_for_sweep)
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
    # Navigation data collection removed; always uses arena
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
    parser_train.add_argument("--activation_fn", type=str, default="gelu", help="Activation function for the model.")
    parser_train.add_argument("--main_net_depth", type=int, default=3, help="Depth of the main network.")
    parser_train.add_argument("--processor_depth", type=int, default=1, help="Depth of the state processors.")
    parser_train.set_defaults(func=handle_train_command)

    # --- Compare Command ---
    parser_compare = subparsers.add_parser(
        "compare",
        help="Compare the performance of different policies or training runs.",
        description="Compare the performance of different policies or training runs.",
    )
    parser_compare.add_argument(
        "policy_names", nargs="*", help="Name(s) of the policy/policies to compare (interactive if omitted)."
    )
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
    parser_sweep.set_defaults(func=handle_sweep_command)

    args = parser.parse_args()

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

    if hasattr(args, "func"):
        args.func(args)
    else:
        # Fallback for interactive mode if func is not set
        if args.command:
            # Manually map command to function if not set by argparse
            command_map = {
                "collect": handle_collect_command,
                "train": handle_train_command,
                "compare": handle_compare_command,
                "sweep": handle_sweep_command,
            }
            if args.command in command_map:
                command_map[args.command](args)
            else:
                print(f"Unknown command: {args.command}")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
