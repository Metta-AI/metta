#!/usr/bin/env python3
"""
Analyze Doxascope Results

Comprehensive analysis of trained doxascope networks.
Provides insights into what patterns the network learned.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from .doxascope_data import Movement
from .doxascope_network import DoxascopeNet


def inspect_data(policy_name: str):
    """Loads and analyzes the preprocessed NPZ data."""
    results_dir = Path(f"doxascope/data/results/{policy_name}")
    data_path = results_dir / "preprocessed_data" / "train_data.npz"

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    X, y = data["X"], data["y"]

    print("\n**Data Overview:**")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Data types: X={X.dtype}, y={y.dtype}")

    print("\n**Memory Vector Stats:**")
    print(f"  - Range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  - Mean: {X.mean():.3f}, Std: {X.std():.3f}")
    if np.isnan(X).any() or np.isinf(X).any():
        print("  - WARNING: Contains NaN or Inf values.")

    print("\n**Movement Class Distribution:**")
    movement_names = ["Stay", "Up", "Down", "Left", "Right"]
    unique_classes, counts = np.unique(y.flatten(), return_counts=True)
    for cls, count in zip(unique_classes, counts, strict=False):
        name = movement_names[cls] if cls < len(movement_names) else f"Unknown({cls})"
        print(f"  - {cls} ({name}): {count:,} samples ({count / len(y.flatten()):.1%})")

    plot_data_inspection(X, y, movement_names, data_path.parent)


def plot_data_inspection(X, y, movement_names, output_dir: Path):
    """Creates and saves visualization plots for the training data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Doxascope Training Data Inspection", fontsize=16)

    # Movement class distribution
    unique_classes, counts = np.unique(y.flatten(), return_counts=True)
    labels = [movement_names[cls] for cls in unique_classes]
    axes[0, 0].bar(labels, counts, color="skyblue")
    axes[0, 0].set_title("Movement Class Distribution")
    axes[0, 0].set_ylabel("Count")

    # Memory vector value distribution
    axes[0, 1].hist(X.flatten(), bins=50, alpha=0.7, color="salmon")
    axes[0, 1].set_title("Memory Vector Value Distribution")
    axes[0, 1].set_xlabel("Value")

    # Mean value per memory dimension
    dim_means = X.mean(axis=0)
    axes[1, 0].plot(dim_means, color="green", alpha=0.8)
    axes[1, 0].set_title("Mean Value per Memory Dimension")
    axes[1, 0].set_xlabel("Dimension")
    axes[1, 0].axvline(x=len(dim_means) // 2, color="red", linestyle="--", alpha=0.7, label="LSTM h/c split")
    axes[1, 0].legend()

    # Std deviation per memory dimension
    dim_stds = X.std(axis=0)
    axes[1, 1].plot(dim_stds, color="purple", alpha=0.8)
    axes[1, 1].set_title("Std Deviation per Memory Dimension")
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].axvline(x=len(dim_stds) // 2, color="red", linestyle="--", alpha=0.7, label="LSTM h/c split")
    axes[1, 1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_dir / "data_inspection.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\n✅ Data inspection plots saved to: {plot_path}")
    plt.close()


def analyze_memory_encoding(policy_name: str):
    """Analyze how well memory vectors encode spatial-temporal information."""
    results_dir = Path(f"doxascope/data/results/{policy_name}")
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found for policy '{policy_name}' at {results_dir}")
        print("👉 Please run the training script first: ")
        print(f"   python -m doxascope.doxascope_train {policy_name}")
        return

    data_path = results_dir / "preprocessed_data" / "training_data.npz"
    model_path = results_dir / "best_model.pth"

    if not data_path.exists() or not model_path.exists():
        print(f"❌ Error: Missing 'training_data.npz' or 'best_model.pth' in {results_dir}")
        return

    # Load data and model
    data = np.load(data_path)
    X, y = data["X"], data["y"]

    # Load the trained model
    checkpoint = torch.load(model_path)
    model = DoxascopeNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    movement_names = ["Stay", "Up", "Down", "Left", "Right"]

    print(f"🔍 Analyzing doxascope memory encoding for policy: '{policy_name}'")
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Analyze memory structure
    analyze_memory_structure(X, y, movement_names)

    # Analyze attention patterns (if available)
    analyze_prediction_confidence(model, X, y, movement_names)

    # PCA analysis
    analyze_pca_components(X, y, movement_names)

    print("\n✅ Analysis complete!")


def analyze_memory_structure(X, y, movement_names):
    """Analyze the structure of memory vectors."""
    print("\n🧠 Memory Vector Analysis")
    print("=" * 40)

    # Split hidden and cell states
    hidden_dim = X.shape[1] // 2
    hidden_states = X[:, :hidden_dim]
    cell_states = X[:, hidden_dim:]

    print(f"Hidden state shape: {hidden_states.shape}")
    print(f"Cell state shape: {cell_states.shape}")

    # Analyze by movement type
    for i, movement in enumerate(movement_names):
        mask = y == i
        if not np.any(mask):
            continue

        h_mean = np.mean(hidden_states[mask], axis=0)
        c_mean = np.mean(cell_states[mask], axis=0)

        print(f"\n{movement}:")
        print(f"  Hidden state mean: {np.mean(h_mean):.4f} (std: {np.std(h_mean):.4f})")
        print(f"  Cell state mean: {np.mean(c_mean):.4f} (std: {np.std(c_mean):.4f})")


def analyze_prediction_confidence(model, X, y, movement_names):
    """Analyze model prediction confidence."""
    print("\n🎯 Prediction Confidence Analysis")
    print("=" * 40)

    with torch.no_grad():
        inputs = torch.FloatTensor(X)
        outputs = model(inputs)

        # We only analyze the first future timestep's prediction for confidence
        num_past = model.config.get("num_past_timesteps", 0)
        if len(outputs) <= num_past:
            print("No future predictions to analyze for confidence.")
            return

        output_t1 = outputs[num_past]
        y_t1 = y[:, num_past]

        probabilities = torch.softmax(output_t1, dim=1)
        predictions = torch.argmax(output_t1, dim=1)

    # Overall confidence
    max_probs = torch.max(probabilities, dim=1)[0]
    mean_confidence = torch.mean(max_probs).item()

    print(f"Average prediction confidence: {mean_confidence:.3f}")

    # Confidence by movement type
    for i, movement in enumerate(movement_names):
        mask = y_t1 == i
        if not np.any(mask):
            continue

        movement_confidence = max_probs[mask].mean().item()
        correct_preds = (predictions[mask] == i).float().mean().item()

        print(f"{movement}: {movement_confidence:.3f} confidence, {correct_preds:.3f} accuracy")

    # Confidence of correct vs incorrect predictions
    correct_mask = predictions.numpy() == y_t1
    correct_confidence = max_probs[correct_mask].mean().item()
    incorrect_confidence = max_probs[~correct_mask].mean().item()

    print(f"\nCorrect predictions: {correct_confidence:.3f} confidence")
    print(f"Incorrect predictions: {incorrect_confidence:.3f} confidence")


def analyze_pca_components(X, y, movement_names):
    """Analyze PCA components of memory vectors."""
    print("\n📐 PCA Analysis")
    print("=" * 40)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # How much variance is captured by top components?
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find how many components capture 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    print(f"Components for 95% variance: {n_components_95} / {X.shape[1]}")
    print(f"Compression ratio: {n_components_95 / X.shape[1] * 100:.1f}%")

    # Top components
    print("\nTop 5 components variance:")
    for i in range(5):
        print(f"  PC{i + 1}: {pca.explained_variance_ratio_[i]:.4f}")

    # Analyze movement separability in PCA space
    print("\nMovement separation in PCA space (first 3 components):")

    for i, movement in enumerate(movement_names):
        mask = y == i
        if not np.any(mask):
            continue

        movement_pca = X_pca[mask, :3]  # First 3 components
        centroid = np.mean(movement_pca, axis=0)

        print(f"{movement}: centroid at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")


def analyze_sweep_results(results_path: Path):
    """Analyze the results of a hyperparameter sweep from a JSON file."""
    if not results_path.exists():
        print(f"❌ Error: Sweep results file not found at {results_path}")
        return

    print(f"📈 Analyzing sweep results from: {results_path.name}")
    print("=" * 50)

    with open(results_path, "r") as f:
        results = json.load(f)

    successful_results = [r for r in results if r.get("success", True)]
    failed_results = [r for r in results if not r.get("success", True)]

    print(f"   Total configs tested: {len(results)}")
    print(f"   Successful runs: {len(successful_results)}")
    print(f"   Failed runs: {len(failed_results)}")

    if not successful_results:
        print("\n   No successful runs to analyze.")
        return

    # Sort by test accuracy
    sorted_results = sorted(successful_results, key=lambda x: x["test_accuracy"], reverse=True)

    # Top 5 results
    print("\n🏆 Top 5 Performing Configurations:")
    for i, result in enumerate(sorted_results[:5]):
        name = result.get("name", f"Config {i + 1}")
        test_acc = result["test_accuracy"]
        print(f"   {i + 1}. {name}: {test_acc:.2f}%")
        config_str = ", ".join([f"{k}={v}" for k, v in result["config"].items()])
        print(f"      Config: {config_str}")

    # Parameter importance analysis
    print("\n🔍 Parameter Impact Analysis:")
    analyze_parameter_importance(successful_results)

    print(f"   Test Accuracy (t+1): {sorted_results[0]['test_accuracy']:.2f}%")


def analyze_parameter_importance(results):
    """Helper to analyze which parameters correlate with performance."""
    param_performance = {}

    for result in results:
        for param, value in result["config"].items():
            if param not in param_performance:
                param_performance[param] = {}
            if value not in param_performance[param]:
                param_performance[param][value] = []
            param_performance[param][value].append(result["test_accuracy"])

    for param, values_dict in param_performance.items():
        if len(values_dict) > 1:
            try:
                value_means = {v: np.mean(accs) for v, accs in values_dict.items()}
                sorted_values = sorted(value_means.items(), key=lambda item: item[1], reverse=True)
                best_value, best_acc = sorted_values[0]
                worst_value, worst_acc = sorted_values[-1]
                impact = best_acc - worst_acc

                print(f"   - {param}: {impact:.2f}% impact")
                print(f"     Best:  {best_value} ({best_acc:.2f}%)")
                print(f"     Worst: {worst_value} ({worst_acc:.2f}%)")
            except (ValueError, TypeError):
                print(f"   - Could not analyze parameter '{param}' (likely non-numeric values).")


def overlay_plots(policy_names: list, output_dir: Path):
    """Generate a plot comparing the multistep accuracy of multiple trained models."""
    plt.figure(figsize=(12, 8))
    plt.title("Multistep Accuracy Comparison")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Timestep")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    all_have_data = True
    max_label_count = 0
    plot_data = []

    for policy_name in policy_names:
        results_path = output_dir / policy_name / "analysis_results.json"
        if not results_path.exists():
            print(f"⚠️ Analysis results not found for policy '{policy_name}'. Skipping.")
            all_have_data = False
            continue

        with open(results_path, "r") as f:
            data = json.load(f)

        acc_per_step = data.get("test_acc_per_step")
        if not acc_per_step:
            print(f"⚠️ No accuracy data found for policy '{policy_name}'. Skipping.")
            all_have_data = False
            continue

        num_past = data.get("num_past_timesteps", 0)
        num_future = data.get("num_future_timesteps", 0)
        past_steps = list(range(-num_past, 0)) if num_past > 0 else []
        future_steps = list(range(1, num_future + 1))
        steps = past_steps + future_steps
        step_labels = [str(s) for s in steps]

        if len(step_labels) > max_label_count:
            max_label_count = len(step_labels)
            plt.xticks(ticks=range(len(step_labels)), labels=step_labels)

        plot_data.append({"labels": step_labels, "accuracies": acc_per_step, "policy": policy_name})

    if not all_have_data:
        print("\nNote: Not all policies had data. The plot may be incomplete.")

    for data in plot_data:
        plt.plot(data["labels"], data["accuracies"], marker="o", linestyle="-", label=data["policy"])

    plt.legend()
    plt.tight_layout()

    overlay_output_dir = output_dir / "overlays"
    overlay_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = overlay_output_dir / "multistep_accuracy_overlay.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\n✅ Overlay plot saved to: {output_path}")


def plot_training_curves(history: Dict[str, Any], output_dir: Path):
    """Plots training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc (avg)")
    plt.plot(history["val_acc"], label="Val Acc (avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    plt.close()


def plot_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, movement_names: List[str], timestep_str: str, output_dir: Path
):
    """Plot the confusion matrix."""
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=movement_names,
        yticklabels=movement_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(f"Confusion Matrix ({timestep_str})")
    plt.savefig(output_dir / f"confusion_matrix_{timestep_str}.png")
    plt.close()


def plot_multistep_accuracy(test_acc_per_step: List[float], model_config: Dict[str, Any], output_dir: Path):
    """Plots test accuracy for each future timestep."""
    num_past = model_config.get("num_past_timesteps", 0)
    num_future = model_config.get("num_future_timesteps", 1)

    if num_past + num_future == 0:
        return

    past_steps = list(range(-num_past, 0)) if num_past > 0 else []
    future_steps = list(range(1, num_future + 1))
    steps = past_steps + future_steps
    step_labels = [str(s) for s in steps]

    if len(step_labels) != len(test_acc_per_step):
        print(
            f"Warning: Mismatch between number of labels ({len(step_labels)}) and accuracies ({len(test_acc_per_step)}). Skipping plot."
        )
        return

    plt.figure(figsize=(10, 6))
    plt.plot(step_labels, test_acc_per_step, marker="o", linestyle="-")
    plt.title("Test Accuracy per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "multistep_accuracy.png")
    plt.close()


def run_analysis(history: Dict, results: Dict, output_dir: Path):
    """Runs all analyses and generates plots."""
    print("\n--- Running Final Analysis ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curves
    plot_training_curves(history, output_dir)

    # Plot multistep accuracy
    model_config = results.get("model_config", {})
    test_acc_per_step = results.get("test_acc_per_step", [])
    if test_acc_per_step:
        plot_multistep_accuracy(test_acc_per_step, model_config, output_dir)

    # Plot confusion matrices
    predictions = results.get("predictions", [])
    targets = results.get("targets", [])
    num_past = model_config.get("num_past_timesteps", 0)
    num_future = model_config.get("num_future_timesteps", 0)
    movement_names = [m.name for m in Movement]

    if num_past > 0 and len(predictions) > 0:
        # Furthest past timestep
        idx = 0
        timestep_str = f"t-{num_past}"
        plot_confusion_matrix(targets[idx], predictions[idx], movement_names, timestep_str, output_dir)

    if num_future > 0 and len(predictions) > num_past:
        # Furthest future timestep
        idx = num_past + num_future - 1
        timestep_str = f"t+{num_future}"
        plot_confusion_matrix(targets[idx], predictions[idx], movement_names, timestep_str, output_dir)


def main():
    """Main CLI entrypoint for analysis."""
    parser = argparse.ArgumentParser(description="Doxascope Analysis Tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'inspect' command
    parser_inspect = subparsers.add_parser("inspect", help="Inspect preprocessed data.")
    parser_inspect.add_argument("policy_name", help="Name of the policy to inspect.")

    # 'encoding' command
    parser_encoding = subparsers.add_parser("encoding", help="Analyze memory encoding.")
    parser_encoding.add_argument("policy_name", help="Name of the policy to analyze.")

    # 'sweep' command
    parser_sweep = subparsers.add_parser("sweep", help="Analyze sweep results.")
    parser_sweep.add_argument("results_path", type=Path, help="Path to the sweep results JSON file.")

    # 'overlay' command
    parser_overlay = subparsers.add_parser("overlay", help="Overlay multistep accuracy plots from multiple policies.")
    parser_overlay.add_argument("policy_names", nargs="+", help="List of policy names to include in the overlay plot.")
    parser_overlay.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doxascope/data/results"),
        help="Directory to save the combined plot.",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_data(args.policy_name)
    elif args.command == "encoding":
        analyze_memory_encoding(args.policy_name)
    elif args.command == "sweep":
        analyze_sweep_results(args.results_path)
    elif args.command == "overlay":
        overlay_plots(args.policy_names, args.output_dir)


if __name__ == "__main__":
    main()
