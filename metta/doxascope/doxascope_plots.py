"""
Doxascope Analysis Plots

Functions to produce predictions, and generate plots.
Includes per-timestep accuracy heatmaps where color = accuracy and
cell annotation = sample count for that relative location.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .doxascope_data import (
    get_class_id_to_pos_map,
    get_num_classes_for_manhattan_distance,
    get_num_classes_for_quadrant_granularity,
    pos_to_quadrant_class_id,
)

if TYPE_CHECKING:
    from .doxascope_network import DoxascopeNet


def get_predictions(model: "DoxascopeNet", X_test: np.ndarray, device: str):
    """Generates per-head predictions and probabilities for X_test."""
    X_test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
    predicted_indices = [torch.argmax(o, dim=1).cpu().numpy() for o in outputs]
    probabilities = [torch.softmax(o, dim=1).cpu().numpy() for o in outputs]
    return predicted_indices, probabilities


def plot_training_history(history: dict, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

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
    if timesteps:
        plt.xticks(timesteps)
    plt.savefig(output_path)
    plt.close()


def _per_class_accuracy_and_counts(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class accuracy and counts for a single head.

    Returns:
        acc_per_class: float array [num_classes]
        count_per_class: int array [num_classes]
    """
    count_per_class = np.bincount(y_true, minlength=num_classes)
    correct_mask = (y_true == y_pred).astype(np.int32)
    correct_per_class = np.bincount(y_true, weights=correct_mask, minlength=num_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_per_class = np.where(count_per_class > 0, correct_per_class / count_per_class, 0.0)
    return acc_per_class.astype(np.float32), count_per_class.astype(np.int64)


def plot_accuracy_heatmaps_per_timestep(
    model: "DoxascopeNet",
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    output_dir: Path,
):
    """Generate a heatmap per timestep head where color=accuracy and text=count.
    For head with timestep k, we build a (2|k|+1)×(2|k|+1) grid over (dr, dc).
    Non-reachable cells (|dr|+|dc|>|k|) are masked to NaN.
    """
    preds, _ = get_predictions(model, X_test, device)
    timesteps: List[int] = model.head_timesteps  # type: ignore[attr-defined]
    granularity = model.config.get("granularity", "exact")

    for head_idx, k in enumerate(timesteps):
        d = abs(k)
        if d == 0:
            continue

        y_true_head = y_test[:, head_idx]
        y_pred_head = preds[head_idx]

        size = 2 * d + 1
        fig, ax = plt.subplots(figsize=(8, 8))

        if granularity == "exact":
            num_classes = get_num_classes_for_manhattan_distance(d)
            acc_c, cnt_c = _per_class_accuracy_and_counts(y_true_head, y_pred_head, num_classes)
            class_id_to_pos = get_class_id_to_pos_map(d)

            acc_grid = np.full((size, size), np.nan, dtype=np.float32)
            cnt_grid = np.zeros((size, size), dtype=np.int64)

            for class_id, (dr, dc) in class_id_to_pos.items():
                r, c = d + dr, d + dc
                acc_grid[r, c] = acc_c[class_id]
                cnt_grid[r, c] = cnt_c[class_id]

            im = ax.imshow(acc_grid, cmap="viridis", vmin=0.0, vmax=1.0)
            for r in range(size):
                for c in range(size):
                    if not np.isnan(acc_grid[r, c]):
                        ax.text(c, r, str(cnt_grid[r, c]), ha="center", va="center", color="white", fontsize=8)

        elif granularity == "quadrant":
            num_quad_classes = get_num_classes_for_quadrant_granularity(k)
            # y_true_head and y_pred_head are already in quadrant space
            acc_q, cnt_q = _per_class_accuracy_and_counts(y_true_head, y_pred_head, num_quad_classes)

            acc_grid = np.full((size, size), np.nan, dtype=np.float32)
            for r in range(size):
                for c in range(size):
                    dr, dc = r - d, c - d
                    if abs(dr) + abs(dc) <= d:
                        quad_id = pos_to_quadrant_class_id(dr, dc)
                        if quad_id < len(acc_q):
                            acc_grid[r, c] = acc_q[quad_id]

            im = ax.imshow(acc_grid, cmap="viridis", vmin=0.0, vmax=1.0)

            # Annotate quadrants
            for quad_id, count in enumerate(cnt_q):
                if count > 0:
                    # Find all grid cells that belong to this quadrant
                    quad_cells = []
                    for r in range(size):
                        for c in range(size):
                            dr, dc = r - d, c - d
                            if abs(dr) + abs(dc) <= d and pos_to_quadrant_class_id(dr, dc) == quad_id:
                                quad_cells.append((r, c))

                    if not quad_cells:
                        continue

                    # Find a representative center for annotation
                    rows, cols = zip(*quad_cells, strict=True)
                    center_r, center_c = np.mean(rows), np.mean(cols)

                    ax.text(
                        center_c,
                        center_r - 0.15,
                        str(count),
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                        fontsize=10,
                    )
                    ax.text(
                        center_c,
                        center_r + 0.15,
                        f"({acc_q[quad_id]:.1%})",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Draw grid lines between cells
        for i in range(size + 1):
            ax.axvline(i - 0.5, color="gray", linewidth=0.5)
            ax.axhline(i - 0.5, color="gray", linewidth=0.5)

        ax.set_title(f"Accuracy Heatmap (timestep {k}, granularity: {granularity})")
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels([str(i) for i in range(-d, d + 1)])
        ax.set_yticklabels([str(i) for i in range(-d, d + 1)])
        ax.set_xlabel("dc (columns)")
        ax.set_ylabel("dr (rows)")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
        plt.tight_layout()
        sign = "+" if k > 0 else ""
        out_path = output_dir / f"acc_heatmap_t{sign}{k}.png"
        plt.savefig(out_path)
        plt.close(fig)


def _compute_inventory_accuracy_by_time(
    model: "DoxascopeNet",
    test_loader: DataLoader,
    device: str,
) -> Tuple[Dict[int, Tuple[int, int]], np.ndarray]:
    """
    Compute inventory prediction accuracy per time-to-change bucket.

    Returns:
        accuracy_stats: Dict mapping time_to_change -> (correct, total)
        time_to_change: Array of all time_to_change values
    """
    num_location_heads = len(model.head_timesteps)

    all_predictions = []
    all_targets = []
    all_time_to_change = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch["X"].to(device)
            outputs = model(batch_x)

            inv_output = outputs[num_location_heads]
            inv_pred = inv_output.argmax(dim=1).cpu().numpy()

            all_predictions.append(inv_pred)
            all_targets.append(batch["y_inventory"].numpy())
            if "time_to_change" in batch:
                all_time_to_change.append(batch["time_to_change"].numpy())

    if not all_predictions or not all_time_to_change:
        return {}, np.array([])

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    time_to_change = np.concatenate(all_time_to_change, axis=0)

    unique_times = sorted(set(time_to_change))
    accuracy_stats: Dict[int, Tuple[int, int]] = {}

    for t in unique_times:
        mask = time_to_change == t
        preds_t = predictions[mask]
        targets_t = targets[mask]
        correct = (preds_t == targets_t).sum()
        total = len(preds_t)
        accuracy_stats[t] = (int(correct), int(total))

    return accuracy_stats, time_to_change


def plot_inventory_accuracy_by_time_to_change(
    model: "DoxascopeNet",
    test_loader: DataLoader,
    device: str,
    resource_names: List[str],
    output_path: Path,
    baseline_model: Optional["DoxascopeNet"] = None,
):
    """
    Generates a plot of inventory prediction accuracy vs time-to-change.

    This is a single-class prediction task: given a memory vector, predict which
    item will change next. Accuracy = % of samples where predicted class matches true class.

    Expected behavior: higher accuracy for imminent changes (t=1,2,3), lower for distant ones.

    Y-axis: accuracy (%)
    X-axis: timesteps until the predicted change occurs
    """
    if "inventory" not in model.prediction_types:
        return

    # Compute main model accuracy
    accuracy_stats, time_to_change = _compute_inventory_accuracy_by_time(model, test_loader, device)

    if len(time_to_change) == 0:
        print("No time_to_change data available for inventory accuracy plot.")
        return

    unique_times = sorted(set(time_to_change))

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot main model accuracy curve
    times = []
    accuracies = []
    sample_counts = []

    for t in unique_times:
        if t in accuracy_stats:
            correct, total = accuracy_stats[t]
            if total > 0:
                times.append(t)
                accuracies.append(100.0 * correct / total)
                sample_counts.append(total)

    if times:
        plt.plot(
            times,
            accuracies,
            marker="o",
            linestyle="-",
            color="steelblue",
            linewidth=2,
            markersize=6,
            label="Main Model",
        )

    # Plot baseline model accuracy if available
    if baseline_model is not None:
        baseline_stats, _ = _compute_inventory_accuracy_by_time(baseline_model, test_loader, device)

        baseline_times = []
        baseline_accuracies = []

        for t in unique_times:
            if t in baseline_stats:
                correct, total = baseline_stats[t]
                if total > 0:
                    baseline_times.append(t)
                    baseline_accuracies.append(100.0 * correct / total)

        if baseline_times:
            plt.plot(
                baseline_times,
                baseline_accuracies,
                marker="x",
                linestyle="--",
                color="red",
                linewidth=1.5,
                markersize=5,
                label="Baseline (random inputs)",
            )

    num_items = len(resource_names)
    plt.xlabel("Timesteps Until Inventory Change")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Inventory Change Prediction: Accuracy vs Time to Change\n"
        f"(Predicting which of {num_items} items will change next)"
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(loc="best", framealpha=0.9)

    # Limit x-axis to reasonable range (trim sparse high-time regions)
    if times:
        good_times = [t for t, n in zip(times, sample_counts, strict=False) if n >= 10]
        if good_times:
            plt.xlim(0, max(good_times) + 5)

    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Inventory accuracy plot saved to {output_path}")


def generate_all_plots(
    output_dir: Path,
    device: str,
    model: "DoxascopeNet",
    history: Dict,
    test_results: Dict,
    test_loader: DataLoader,
    is_baseline: bool = False,
    resource_names: Optional[List[str]] = None,
    baseline_model: Optional["DoxascopeNet"] = None,
):
    """Generates all analysis plots for a training run."""
    if is_baseline:
        return

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating analysis plots in: {analysis_dir}")

    # Extract test data from loader for heatmaps
    X_test_list, y_location_list = [], []
    for batch in test_loader:
        X_test_list.append(batch["X"].cpu().numpy())
        y_location_list.append(batch["y_location"].cpu().numpy())
    X_test = np.concatenate(X_test_list, axis=0)
    y_location = np.concatenate(y_location_list, axis=0)

    # Plot training history for the main model
    plot_training_history(history, analysis_dir / "training_history.png")

    # Plot multistep accuracy, including baseline if available
    baseline_results = None
    baseline_results_path = output_dir / "test_results_baseline.json"
    if baseline_results_path.exists():
        with open(baseline_results_path, "r") as f:
            baseline_results = json.load(f)
    plot_multistep_accuracy(
        test_results,
        analysis_dir / "multistep_accuracy_comparison.png",
        baseline_results=baseline_results,
    )

    # Plot heatmaps for the main model (location predictions only)
    if "location" in model.prediction_types:
        plot_accuracy_heatmaps_per_timestep(model, X_test, y_location, device, analysis_dir)

    # Plot inventory accuracy by time-to-change
    if "inventory" in model.prediction_types and resource_names:
        plot_inventory_accuracy_by_time_to_change(
            model,
            test_loader,
            device,
            resource_names,
            analysis_dir / "inventory_accuracy_by_time.png",
            baseline_model=baseline_model,
        )

    print(f"Successfully generated plots for run in {output_dir.name}")
