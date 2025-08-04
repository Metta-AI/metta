#!/usr/bin/env python3
"""
Analyze Doxascope Results

Comprehensive analysis of trained doxascope networks.
Provides insights into what patterns the network learned.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .doxascope_data import class_id_to_pos
from .doxascope_network import DoxascopeNet


def load_data_and_model(
    policy_dir: Path,
    device: str,
    model_filename: str = "best_model.pth",
    results_filename: str = "test_results.json",
    history_filename: str = "training_history.csv",
) -> Tuple[DoxascopeNet, np.ndarray, np.ndarray, Dict, Dict]:
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


def get_predictions(model: DoxascopeNet, X_test: np.ndarray, device: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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


def plot_confusion_matrices(y_true: np.ndarray, y_pred: List[np.ndarray], timesteps: List[int], output_dir: Path):
    """
    For each timestep, plots a heatmap of prediction accuracy per grid cell.
    Overlays text indicating the true frequency of each ground truth position.
    """
    for i, timestep in enumerate(timesteps):
        true_labels = y_true[:, i]
        pred_labels = y_pred[i]
        max_dist = abs(timestep)

        # Convert labels to positions
        true_pos = [class_id_to_pos(l, max_dist) for l in true_labels]
        pred_pos = [class_id_to_pos(l, max_dist) for l in pred_labels]

        df = pd.DataFrame(
            {
                "true_dr": [p[0] for p in true_pos],
                "true_dc": [p[1] for p in true_pos],
                "pred_dr": [p[0] for p in pred_pos],
                "pred_dc": [p[1] for p in pred_pos],
            }
        )

        # Calculate true frequencies for each ground truth position
        true_counts = df.groupby(["true_dr", "true_dc"]).size().reset_index()
        true_counts.rename(columns={0: "true_freq"}, inplace=True)

        # Calculate correct prediction counts for each position
        correct_preds = df[(df["true_dr"] == df["pred_dr"]) & (df["true_dc"] == df["pred_dc"])]
        correct_counts = correct_preds.groupby(["true_dr", "true_dc"]).size().reset_index()
        correct_counts.rename(columns={0: "correct_freq"}, inplace=True)

        # Merge true counts and correct counts to calculate accuracy
        accuracy_df = pd.merge(true_counts, correct_counts, on=["true_dr", "true_dc"], how="left")
        accuracy_df["correct_freq"] = accuracy_df["correct_freq"].fillna(0)
        accuracy_df["accuracy"] = (accuracy_df["correct_freq"] / accuracy_df["true_freq"]) * 100

        # Create a grid for the accuracy heatmap
        grid_size = 2 * max_dist + 1
        accuracy_grid = np.full((grid_size, grid_size), -1.0)  # Use -1 for no data

        # Populate the grid with accuracy values
        for _, row in accuracy_df.iterrows():
            r_idx, c_idx = int(row["true_dr"] + max_dist), int(row["true_dc"] + max_dist)
            accuracy_grid[r_idx, c_idx] = row["accuracy"]

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create a masked array to hide cells with no data
        masked_grid = np.ma.masked_where(accuracy_grid == -1, accuracy_grid)

        # To align pixels with grid, we define the extent of the image.
        half_extent = max_dist + 0.5
        extent = (-half_extent, half_extent, -half_extent, half_extent)

        im = ax.imshow(
            masked_grid,
            cmap="viridis",
            vmin=0,
            vmax=100,
            interpolation="none",
            extent=extent,
            origin="lower",  # Puts (0,0) of the array at the bottom-left
        )
        ax.set_facecolor("lightgray")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Predictive Accuracy (%)")

        # Set ticks for the center of the cells
        ticks = range(-max_dist, max_dist + 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Set grid lines to be at the boundaries of the cells
        ax.set_xticks(np.arange(-max_dist, max_dist + 2, 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(-max_dist, max_dist + 2, 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", size=0)  # Hide minor tick marks

        # Add text annotations for true frequency
        for r_idx in range(grid_size):
            for c_idx in range(grid_size):
                r, c = r_idx - max_dist, c_idx - max_dist

                if abs(r) + abs(c) > max_dist:
                    continue

                # Find the frequency for this cell
                count_row = true_counts[(true_counts["true_dr"] == r) & (true_counts["true_dc"] == c)]
                freq = count_row["true_freq"].iloc[0] if not count_row.empty else 0

                # Determine text color based on cell accuracy
                cell_accuracy = accuracy_grid[r_idx, c_idx]
                text_color = "gray"
                if freq > 0:
                    text_color = "white" if cell_accuracy < 60 else "black"

                ax.text(
                    c,  # x-coordinate in data space
                    r,  # y-coordinate in data space
                    str(freq),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    weight="normal" if freq == 0 else "bold",
                )

        plt.title(f"Prediction Accuracy (Color) vs. True Frequency (Number) for Timestep t={timestep}")
        plt.xlabel("Relative Column (dc)")
        plt.ylabel("Relative Row (dr)")
        plt.savefig(output_dir / f"confusion_matrix_t_{timestep}.png")
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
    plot_confusion_matrices(y_test, predicted_indices, test_results["timesteps"], analysis_dir)

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

        baseline_history_path = policy_dir / "training_history_baseline.csv"
        if baseline_history_path.exists():
            baseline_history = pd.read_csv(baseline_history_path).to_dict(orient="list")
            plot_training_history(baseline_history, analysis_dir / "training_history_baseline.png")

    print(f"Successfully generated plots for run {policy_dir.name}")

    # Also generate plots for the baseline model if it exists
    baseline_model_path = policy_dir / "best_model_baseline.pth"
    if baseline_model_path.exists():
        print("\nGenerating analysis plots for baseline model...")
        try:
            (
                model_bl,
                X_test_bl,
                y_test_bl,
                history_bl,
                test_results_bl,
            ) = load_data_and_model(
                policy_dir,
                device,
                model_filename="best_model_baseline.pth",
                results_filename="test_results_baseline.json",
                history_filename="training_history_baseline.csv",
            )
            predicted_indices_bl, _ = get_predictions(model_bl, X_test_bl, device)

            baseline_analysis_dir = analysis_dir / "baseline"
            baseline_analysis_dir.mkdir(exist_ok=True)

            plot_training_history(history_bl, baseline_analysis_dir / "training_history.png")
            plot_multistep_accuracy(test_results_bl, baseline_analysis_dir / "multistep_accuracy.png")
            plot_confusion_matrices(
                y_test_bl, predicted_indices_bl, test_results_bl["timesteps"], baseline_analysis_dir
            )

            print(f"Successfully generated plots for baseline in {baseline_analysis_dir}")

        except FileNotFoundError as e:
            print(f"Could not generate baseline plots: {e}")


def find_latest_run(policy_dir: Path) -> Optional[Path]:
    """Finds the most recent run directory in a policy's result folder."""
    run_dirs = [d for d in policy_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.name)


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


def analyze_label_distribution(y_true: np.ndarray, timesteps: List[int]):
    """Analyzes and prints the distribution of labels, focusing on the 'stay' action."""
    print("\n--- Label Distribution Analysis ---")
    for i, timestep in enumerate(timesteps):
        labels = y_true[:, i]
        max_dist = abs(timestep)

        # Find the class ID for the 'stay' action (dr=0, dc=0)
        try:
            stay_id = class_id_to_pos(0, 0, max_dist)
        except ValueError:
            stay_id = -1  # Impossible for this timestep

        if stay_id != -1:
            stay_count = np.sum(labels == stay_id)
            total_count = len(labels)
            stay_percentage = (stay_count / total_count) * 100
            print(f"Timestep t={timestep}:")
            print(f"  - 'Stay' actions (dr=0, dc=0): {stay_count}/{total_count} ({stay_percentage:.2f}%)")
        else:
            print(f"Timestep t={timestep}: 'Stay' action is not possible.")


def main():
    parser = argparse.ArgumentParser(description="Analyze a trained DoxascopeNet model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Analyze a single run ---
    parser_analyze = subparsers.add_parser("analyze", help="Analyze a single training run.")
    parser_analyze.add_argument("policy_name", type=str, help="Name of the policy to analyze.")
    parser_analyze.add_argument(
        "run_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the specific run to analyze. If omitted, the latest run will be used.",
    )
    parser_analyze.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing the policy subdirectories.",
    )
    parser_analyze.add_argument(
        "--device", type=str, default="auto", help="Device to use for analysis (e.g., 'cpu', 'cuda')."
    )

    # --- Re-analyze all runs for a policy ---
    parser_reanalyze = subparsers.add_parser(
        "reanalyze-all", help="Re-generate analysis plots for all runs of a policy."
    )
    parser_reanalyze.add_argument("policy_name", type=str, help="Name of the policy to re-analyze.")
    parser_reanalyze.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing the policy subdirectories.",
    )
    parser_reanalyze.add_argument(
        "--device", type=str, default="auto", help="Device to use for analysis (e.g., 'cpu', 'cuda')."
    )

    # --- Compare multiple runs or policies ---
    parser_compare = subparsers.add_parser("compare", help="Compare training runs for one or more policies.")
    parser_compare.add_argument("policy_names", nargs="+", help="Name(s) of the policy/policies to compare.")
    parser_compare.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing the policy subdirectories.",
    )

    # --- Describe a dataset ---
    parser_describe = subparsers.add_parser("describe-data", help="Describe the preprocessed dataset for a run.")
    parser_describe.add_argument("policy_name", type=str, help="Name of the policy.")
    parser_describe.add_argument(
        "run_name",
        type=str,
        nargs="?",
        default=None,
        help="Name of the run. If omitted, the latest run is used.",
    )
    parser_describe.add_argument(
        "--data-dir",
        type=Path,
        default=Path("train_dir/doxascope/results"),
        help="Directory containing the policy subdirectories.",
    )
    parser_describe.add_argument(
        "--dataset", type=str, default="test", choices=["train", "test", "val"], help="Which dataset split to describe."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(args, "device") and args.device != "auto":
        device = args.device

    if args.command == "analyze":
        policy_dir = args.data_dir / args.policy_name
        if not policy_dir.is_dir():
            print(f"Error: Policy directory not found at {policy_dir}")
            return

        if args.run_name:
            policy_run_dir = policy_dir / args.run_name
            if not policy_run_dir.is_dir():
                print(f"Error: Run directory not found at {policy_run_dir}")
                return
        else:
            print(f"No run name specified, finding the latest run for policy '{args.policy_name}'...")
            policy_run_dir = find_latest_run(policy_dir)
            if policy_run_dir is None:
                print(f"Error: No runs found for policy '{args.policy_name}'.")
                return
            print(f"Analyzing latest run: {policy_run_dir.name}")

        generate_all_plots(policy_run_dir, device)

    elif args.command == "compare":
        if len(args.policy_names) == 1:
            # Compare all runs for a single policy
            policy_name = args.policy_names[0]
            policy_dir = args.data_dir / policy_name
            compare_runs(policy_dir, policy_dir)
        else:
            # Compare the latest run of multiple policies
            compare_policies(args.policy_names, args.data_dir, args.data_dir)

    elif args.command == "reanalyze-all":
        policy_dir = args.data_dir / args.policy_name
        if not policy_dir.is_dir():
            print(f"Error: Policy directory not found at {policy_dir}")
            return
        for run_dir in sorted(policy_dir.iterdir()):
            if run_dir.is_dir():
                generate_all_plots(run_dir, device)

    elif args.command == "describe-data":
        policy_dir = args.data_dir / args.policy_name
        if args.run_name:
            run_dir = policy_dir / args.run_name
        else:
            run_dir = find_latest_run(policy_dir)

        if run_dir is None or not run_dir.exists():
            print(f"Error: Could not find run directory for {args.policy_name}")
            return

        data_path = run_dir / "preprocessed_data" / f"{args.dataset}.npz"
        if not data_path.exists():
            print(f"Error: {args.dataset}.npz not found in {run_dir}")
            return

        data = np.load(data_path)
        y_true = data["y"]

        # We need the timesteps, which are stored in the model config or test_results
        results_path = run_dir / "test_results.json"
        if not results_path.exists():
            print("Warning: test_results.json not found. Cannot determine timesteps.")
            # As a fallback, try to infer from y_true shape if it's 2D
            if y_true.ndim == 2:
                num_timesteps = y_true.shape[1]
                # This is a guess; we don't know if they are past or future.
                # Assuming future for now as it's more common.
                timesteps = list(range(1, num_timesteps + 1))
                print(f"Inferred {num_timesteps} timesteps. Assuming future timesteps: {timesteps}")
            else:
                print("Cannot determine timesteps for analysis.")
                return
        else:
            with open(results_path, "r") as f:
                results = json.load(f)
            timesteps = results["timesteps"]

        analyze_label_distribution(y_true, timesteps)


if __name__ == "__main__":
    main()
