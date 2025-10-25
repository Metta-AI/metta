"""
Doxascope Analysis Utilities

"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .doxascope_network import DoxascopeNet


def load_data_and_model(
    policy_dir: Path,
    device: str,
    model_filename: str = "best_model.pth",
    results_filename: str = "test_results.json",
    history_filename: str = "training_history.csv",
):
    """Loads the model, data, and results for a given policy run directory."""
    model_path = policy_dir / model_filename
    test_data_path = policy_dir / "preprocessed_data" / "test.npz"
    history_path = policy_dir / history_filename
    test_results_path = policy_dir / results_filename

    # Load the full model checkpoint.
    # The state_dict file is kept for manual/external analysis but not used here.
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model checkpoint from {model_path}: {e}")
        raise

    model = DoxascopeNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    test_data = np.load(test_data_path)
    X_test, y_test = test_data["X"], test_data["y"]

    history = pd.read_csv(history_path).to_dict(orient="list")
    with open(test_results_path, "r") as f:
        test_results = json.load(f)

    return model, X_test, y_test, history, test_results


def compare_policies(policy_names: List[str], data_dir: Path, output_dir: Path):
    latest_run_results = {}
    for name in policy_names:
        policy_dir = data_dir / name
        # Find latest run by presence of test_results.json
        run_dirs = [d for d in policy_dir.iterdir() if d.is_dir() and (d / "test_results.json").exists()]
        if not run_dirs:
            print(f"Warning: No runs found for policy '{name}'. Skipping.")
            continue
        latest = max(run_dirs, key=lambda p: p.name)
        with open(latest / "test_results.json", "r") as f:
            latest_run_results[name] = (json.load(f), latest)

    if not latest_run_results:
        print("No valid runs found to compare.")
        return

    plt.figure(figsize=(12, 8))
    for policy_name, (results, run_dir) in latest_run_results.items():
        if "timesteps" in results and "test_accuracy_per_step" in results:
            train_data_path = run_dir / "preprocessed_data" / "train.npz"
            data_size_str = "N/A"
            if train_data_path.exists():
                size_bytes = train_data_path.stat().st_size
                data_size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
            plt.plot(
                results["timesteps"],
                results["test_accuracy_per_step"],
                marker="o",
                linestyle="-",
                label=f"Policy: {policy_name} ({data_size_str})",
            )

    plt.title("Latest Run Accuracy Comparison")
    plt.xlabel("Timestep Relative to Present")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Consolidate all timesteps to set shared x-axis ticks
    all_timesteps = set()
    for results, _ in latest_run_results.values():
        if "timesteps" in results:
            all_timesteps.update(results["timesteps"])
    if all_timesteps:
        plt.xticks(sorted(list(all_timesteps)))

    plt.tight_layout()
    safe_name = "_vs_".join(p.replace("/", "_") for p in policy_names)
    comparison_plot_path = output_dir / f"comparison_{safe_name}.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Policy comparison plot saved to {comparison_plot_path}")


def compare_runs(run_dirs: List[Path], policy_name: str, output_dir: Path):
    run_results = {}
    # Sort runs from newest to oldest to find the latest baseline
    sorted_run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    baseline_results = None
    # Find the first available baseline from the sorted list of runs
    for run_dir in sorted_run_dirs:
        baseline_path = run_dir / "test_results_baseline.json"
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                baseline_results = json.load(f)
            print(f"Using baseline from run: {run_dir.name}")
            break

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "test_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                run_results[run_dir.name] = json.load(f)

    if not run_results:
        print("No completed runs found in the selected directories.")
        return

    plt.figure(figsize=(12, 8))
    for run_name, results in sorted(run_results.items()):
        if "timesteps" in results and "test_accuracy_per_step" in results:
            # Find the full path for the run_name to get data size
            matching_run_dir: Optional[Path] = next((rd for rd in run_dirs if rd.name == run_name), None)
            data_size_str = "N/A"
            if matching_run_dir:
                train_data_path = matching_run_dir / "preprocessed_data" / "train.npz"
                if train_data_path.exists():
                    size_bytes = train_data_path.stat().st_size
                    data_size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
            plt.plot(
                results["timesteps"],
                results["test_accuracy_per_step"],
                marker="o",
                linestyle="-",
                label=f"Run: {run_name} ({data_size_str})",
            )

    if baseline_results:
        plt.plot(
            baseline_results["timesteps"],
            baseline_results["test_accuracy_per_step"],
            marker="x",
            linestyle="--",
            label="Random Baseline (latest available)",
        )

    plt.title(f"Multistep Accuracy Comparison for Policy: {policy_name}")
    plt.xlabel("Timestep Relative to Present")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Consolidate all timesteps to set shared x-axis ticks
    all_timesteps = set()
    for results in run_results.values():
        if "timesteps" in results:
            all_timesteps.update(results["timesteps"])
    if baseline_results and "timesteps" in baseline_results:
        all_timesteps.update(baseline_results["timesteps"])
    if all_timesteps:
        plt.xticks(sorted(list(all_timesteps)))

    plt.tight_layout()
    comparison_plot_path = output_dir / "comparison_plot.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Comparison plot saved to {comparison_plot_path}")
