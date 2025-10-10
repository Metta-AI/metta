#!/usr/bin/env python3
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

from .doxascope_data import get_class_id_to_pos_map, get_num_classes_for_manhattan_distance

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

    for head_idx, k in enumerate(timesteps):
        d = abs(k)
        if d == 0:
            continue
        num_classes = get_num_classes_for_manhattan_distance(d)
        acc_c, cnt_c = _per_class_accuracy_and_counts(y_test[:, head_idx], preds[head_idx], num_classes)

        # Build grid
        size = 2 * d + 1
        acc_grid = np.full((size, size), np.nan, dtype=np.float32)
        cnt_grid = np.zeros((size, size), dtype=np.int64)
        class_id_to_pos = get_class_id_to_pos_map(d)
        # Map center at (d, d) representing (dr=0, dc=0)
        for class_id, (dr, dc) in class_id_to_pos.items():
            r = d + dr
            c = d + dc
            acc_grid[r, c] = acc_c[class_id]
            cnt_grid[r, c] = cnt_c[class_id]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(acc_grid, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Accuracy Heatmap (timestep {k})")
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels([str(i) for i in range(-d, d + 1)])
        ax.set_yticklabels([str(i) for i in range(-d, d + 1)])
        ax.set_xlabel("dc (columns)")
        ax.set_ylabel("dr (rows)")

        # Annotate counts
        for r in range(size):
            for c in range(size):
                if np.isnan(acc_grid[r, c]):
                    continue
                ax.text(c, r, str(cnt_grid[r, c]), ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
        plt.tight_layout()
        sign = "+" if k > 0 else ""
        out_path = output_dir / f"acc_heatmap_t{sign}{k}.png"
        plt.savefig(out_path)
        plt.close(fig)


def generate_all_plots(
    output_dir: Path,
    device: str,
    model: "DoxascopeNet",
    history: Dict,
    test_results: Dict,
    test_loader: DataLoader,
    is_baseline: bool = False,
):
    """Generates all analysis plots for a training run."""
    if is_baseline:
        return

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating analysis plots in: {analysis_dir}")

    # Extract test data from loader for heatmaps
    X_test_list, y_test_list = [], []
    for batch_x, batch_y in test_loader:
        X_test_list.append(batch_x.cpu().numpy())
        y_test_list.append(batch_y.cpu().numpy())
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

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

    # Plot heatmaps for the main model
    plot_accuracy_heatmaps_per_timestep(model, X_test, y_test, device, analysis_dir)
    print(f"Successfully generated plots for run in {output_dir.name}")
