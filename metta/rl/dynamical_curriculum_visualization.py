"""
Visualization module for dynamical curriculum analysis.

This module provides comprehensive plotting capabilities for analyzing
the results of the dynamical curriculum learning experiments.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Try to import wandb for logging
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Visualizations will only be saved locally.")


class DynamicalCurriculumVisualizer:
    """Visualization class for dynamical curriculum analysis results."""

    @staticmethod
    def create_comprehensive_visualization(
        results: Dict[str, Dict],
        output_path: str = "dynamical_curriculum_analysis.png",
        figsize: tuple = (20, 12),
        log_to_wandb: bool = True,
    ):
        """
        Create a comprehensive visualization of all curriculum results.

        Args:
            results: Dictionary containing results for each curriculum
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
            log_to_wandb: Whether to log the plot to WandB
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Performance trajectories (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        DynamicalCurriculumVisualizer._plot_performance_trajectories(results, ax1)

        # 2. Learning efficiency comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        DynamicalCurriculumVisualizer._plot_efficiency_comparison(results, ax2)

        # 3. Time to threshold comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        DynamicalCurriculumVisualizer._plot_time_to_threshold(results, ax3)

        # 4. Final performance comparison (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        DynamicalCurriculumVisualizer._plot_final_performance(results, ax4)

        # 5. Sampling pattern heatmap (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        DynamicalCurriculumVisualizer._plot_sampling_patterns(results, ax5)

        # 6. Performance variance over time (bottom row, spans 2 columns)
        ax6 = fig.add_subplot(gs[2, :2])
        DynamicalCurriculumVisualizer._plot_performance_variance(results, ax6)

        # 7. Curriculum regret analysis or grid search (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        if "grid_search" in results:
            DynamicalCurriculumVisualizer._plot_grid_search_heatmap(results["grid_search"], ax7)
        else:
            DynamicalCurriculumVisualizer._plot_curriculum_regret(results, ax7)

        # Add title
        fig.suptitle("Dynamical Curriculum Learning Analysis", fontsize=16, fontweight="bold")

        # Save locally
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive visualization saved to: {output_path}")

        # Log to WandB if available and requested
        DynamicalCurriculumVisualizer._log_plot_to_wandb(fig, "comprehensive_analysis", log_to_wandb)

        plt.close()

    @staticmethod
    def _log_plot_to_wandb(fig: plt.Figure, plot_name: str, log_to_wandb: bool = True):
        """Helper function to log a plot to WandB."""
        if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({plot_name: wandb.Image(fig)})
            print(f"{plot_name} logged to WandB")

    @staticmethod
    def _plot_performance_trajectories(results: Dict[str, Dict], ax: plt.Axes):
        """Plot performance trajectories over time."""
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, (curriculum_name, result) in enumerate(results.items()):
            if curriculum_name == "grid_search":
                continue  # Skip grid search in this plot

            performance_history = result["performance_history"]
            epochs = range(len(performance_history))

            ax.plot(
                epochs,
                performance_history,
                label=curriculum_name.replace("_", " ").title(),
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Average Performance")
        ax.set_title("Performance Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add success threshold line
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="Success Threshold")

    @staticmethod
    def _plot_efficiency_comparison(results: Dict[str, Dict], ax: plt.Axes):
        """Plot learning efficiency comparison."""
        curricula = [name for name in results.keys() if name != "grid_search"]
        efficiencies = [results[curriculum]["learning_efficiency"] for curriculum in curricula]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(curricula, efficiencies, color=colors[: len(curricula)], alpha=0.7)

        # Add value labels on bars
        for bar, efficiency in zip(bars, efficiencies, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(efficiencies) * 0.01,
                f"{efficiency:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Learning Efficiency (AUC)")
        ax.set_title("Learning Efficiency Comparison")
        ax.tick_params(axis="x", rotation=45)

    @staticmethod
    def _plot_time_to_threshold(results: Dict[str, Dict], ax: plt.Axes):
        """Plot time to threshold comparison."""
        curricula = [name for name in results.keys() if name != "grid_search"]
        times = []
        labels = []

        for curriculum in curricula:
            time_to_thresh = results[curriculum]["time_to_threshold"]
            if time_to_thresh is not None:
                times.append(time_to_thresh)
                labels.append(curriculum.replace("_", " ").title())
            else:
                times.append(0)  # Will be filtered out
                labels.append(curriculum.replace("_", " ").title() + " (∞)")

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(labels, times, color=colors[: len(curricula)], alpha=0.7)

        # Add value labels on bars
        for bar, time_val in zip(bars, times, strict=False):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(times) * 0.01,
                    f"{time_val}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_ylabel("Epochs to Threshold")
        ax.set_title("Time to Success Threshold")
        ax.tick_params(axis="x", rotation=45)

    @staticmethod
    def _plot_final_performance(results: Dict[str, Dict], ax: plt.Axes):
        """Plot final performance comparison."""
        curricula = [name for name in results.keys() if name != "grid_search"]
        final_perfs = [results[curriculum]["final_performance"] for curriculum in curricula]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(curricula, final_perfs, color=colors[: len(curricula)], alpha=0.7)

        # Add value labels on bars
        for bar, perf in zip(bars, final_perfs, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(final_perfs) * 0.01,
                f"{perf:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Final Performance")
        ax.set_title("Final Performance Comparison")
        ax.tick_params(axis="x", rotation=45)

        # Add success threshold line
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7)

    @staticmethod
    def _plot_sampling_patterns(results: Dict[str, Dict], ax: plt.Axes):
        """Plot sampling pattern heatmap for the learning progress curriculum."""
        if "learning_progress" not in results:
            ax.text(0.5, 0.5, "No learning progress data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Sampling Patterns")
            return

        # Get sampling history for learning progress curriculum
        sampling_history = results["learning_progress"]["sampling_history"]

        # Convert to matrix (epochs x tasks)
        num_epochs = len(sampling_history)
        num_tasks = len(sampling_history[0])

        sampling_matrix = np.zeros((num_epochs, num_tasks))
        for epoch in range(num_epochs):
            for task_idx, (_task, prob) in enumerate(sampling_history[epoch].items()):
                sampling_matrix[epoch, task_idx] = prob

        # Plot heatmap
        im = ax.imshow(sampling_matrix.T, aspect="auto", cmap="viridis")

        # Set labels
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Task")
        ax.set_title("Learning Progress Sampling Patterns")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Sampling Probability")

    @staticmethod
    def _plot_performance_variance(results: Dict[str, Dict], ax: plt.Axes):
        """Plot performance variance over time."""
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, (curriculum_name, result) in enumerate(results.items()):
            if curriculum_name == "grid_search":
                continue

            # Calculate variance across tasks over time
            if "final_performance_vector" in result:
                # For now, just show the final performance vector
                final_perfs = result["final_performance_vector"]
                task_indices = range(len(final_perfs))

                ax.plot(
                    task_indices,
                    final_perfs,
                    label=curriculum_name.replace("_", " ").title(),
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                    marker="o",
                    markersize=4,
                )

        ax.set_xlabel("Task Index")
        ax.set_ylabel("Final Performance")
        ax.set_title("Final Performance by Task")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_curriculum_regret(results: Dict[str, Dict], ax: plt.Axes):
        """Plot curriculum regret analysis."""
        if "oracle" not in results:
            ax.text(0.5, 0.5, "No oracle data for regret analysis", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Curriculum Regret")
            return

        # Calculate regret relative to oracle
        oracle_performance = results["oracle"]["performance_history"]
        curricula = [name for name in results.keys() if name not in ["oracle", "grid_search"]]

        colors = ["#1f77b4", "#ff7f0e"]

        for i, curriculum_name in enumerate(curricula):
            curriculum_performance = results[curriculum_name]["performance_history"]

            # Calculate cumulative regret
            regret = []
            cumulative_regret = 0
            for epoch in range(len(oracle_performance)):
                if epoch < len(curriculum_performance):
                    regret_epoch = max(0, oracle_performance[epoch] - curriculum_performance[epoch])
                    cumulative_regret += regret_epoch
                    regret.append(cumulative_regret)

            ax.plot(
                range(len(regret)),
                regret,
                label=curriculum_name.replace("_", " ").title(),
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title("Curriculum Regret vs Oracle")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_grid_search_heatmap(grid_results: Dict[Tuple[float, float], Dict[str, float]], ax: plt.Axes):
        """Plot grid search results as a heatmap."""
        if not grid_results:
            ax.text(0.5, 0.5, "No grid search data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Grid Search Results")
            return

        # Extract unique parameter values
        ema_timescales = sorted(list(set(param[0] for param in grid_results.keys())))
        progress_smoothings = sorted(list(set(param[1] for param in grid_results.keys())))

        # Create efficiency matrix
        efficiency_matrix = np.zeros((len(progress_smoothings), len(ema_timescales)))

        for i, smoothing in enumerate(progress_smoothings):
            for j, ema in enumerate(ema_timescales):
                if (ema, smoothing) in grid_results:
                    efficiency_matrix[i, j] = grid_results[(ema, smoothing)]["learning_efficiency"]

        # Plot heatmap
        im = ax.imshow(efficiency_matrix, cmap="viridis", aspect="auto")

        # Set labels
        ax.set_xlabel("EMA Timescale")
        ax.set_ylabel("Progress Smoothing")
        ax.set_title("Grid Search: Learning Efficiency")

        # Set tick labels
        ax.set_xticks(range(len(ema_timescales)))
        ax.set_xticklabels([f"{ema:.1e}" for ema in ema_timescales], rotation=45)
        ax.set_yticks(range(len(progress_smoothings)))
        ax.set_yticklabels([f"{smoothing:.3f}" for smoothing in progress_smoothings])

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Learning Efficiency")

        # Add best parameter annotation
        best_params = max(grid_results.items(), key=lambda x: x[1]["learning_efficiency"])
        best_ema_idx = ema_timescales.index(best_params[0][0])
        best_smoothing_idx = progress_smoothings.index(best_params[0][1])
        ax.plot(best_ema_idx, best_smoothing_idx, "r*", markersize=15, label="Best")
        ax.legend()

    @staticmethod
    def create_grid_search_visualization(
        grid_results: Dict[Tuple[float, float], Dict[str, float]],
        output_path: str = "grid_search_analysis.png",
        figsize: tuple = (15, 10),
        log_to_wandb: bool = True,
    ):
        """
        Create a comprehensive visualization of grid search results.

        Args:
            grid_results: Dictionary mapping (ema_timescale, progress_smoothing) to metrics
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
            log_to_wandb: Whether to log the plot to WandB
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Extract parameters and metrics
        ema_timescales = sorted(list(set(param[0] for param in grid_results.keys())))
        progress_smoothings = sorted(list(set(param[1] for param in grid_results.keys())))

        # Create efficiency heatmap
        efficiency_matrix = np.zeros((len(ema_timescales), len(progress_smoothings)))
        for i, ema in enumerate(ema_timescales):
            for j, smoothing in enumerate(progress_smoothings):
                if (ema, smoothing) in grid_results:
                    efficiency_matrix[i, j] = grid_results[(ema, smoothing)]["learning_efficiency"]

        # Plot 1: Efficiency heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(efficiency_matrix, cmap="viridis", aspect="auto")
        ax1.set_xticks(range(len(progress_smoothings)))
        ax1.set_xticklabels([f"{s:.3f}" for s in progress_smoothings], rotation=45)
        ax1.set_yticks(range(len(ema_timescales)))
        ax1.set_yticklabels([f"{e:.5f}" for e in ema_timescales])
        ax1.set_xlabel("Progress Smoothing")
        ax1.set_ylabel("EMA Timescale")
        ax1.set_title("Learning Efficiency Heatmap")
        plt.colorbar(im1, ax=ax1)

        # Create final performance heatmap
        performance_matrix = np.zeros((len(ema_timescales), len(progress_smoothings)))
        for i, ema in enumerate(ema_timescales):
            for j, smoothing in enumerate(progress_smoothings):
                if (ema, smoothing) in grid_results:
                    performance_matrix[i, j] = grid_results[(ema, smoothing)]["final_performance"]

        # Plot 2: Final performance heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(performance_matrix, cmap="plasma", aspect="auto")
        ax2.set_xticks(range(len(progress_smoothings)))
        ax2.set_xticklabels([f"{s:.3f}" for s in progress_smoothings], rotation=45)
        ax2.set_yticks(range(len(ema_timescales)))
        ax2.set_yticklabels([f"{e:.5f}" for e in ema_timescales])
        ax2.set_xlabel("Progress Smoothing")
        ax2.set_ylabel("EMA Timescale")
        ax2.set_title("Final Performance Heatmap")
        plt.colorbar(im2, ax=ax2)

        # Plot 3: Parameter sensitivity - EMA timescale
        ax3 = axes[1, 0]
        ema_efficiencies = []
        for ema in ema_timescales:
            efficiencies = [
                grid_results[(ema, smoothing)]["learning_efficiency"]
                for smoothing in progress_smoothings
                if (ema, smoothing) in grid_results
            ]
            ema_efficiencies.append(np.mean(efficiencies) if efficiencies else 0)

        ax3.plot(ema_timescales, ema_efficiencies, marker="o", linewidth=2)
        ax3.set_xlabel("EMA Timescale")
        ax3.set_ylabel("Average Learning Efficiency")
        ax3.set_title("EMA Timescale Sensitivity")
        ax3.set_xscale("log")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Parameter sensitivity - Progress smoothing
        ax4 = axes[1, 1]
        smoothing_efficiencies = []
        for smoothing in progress_smoothings:
            efficiencies = [
                grid_results[(ema, smoothing)]["learning_efficiency"]
                for ema in ema_timescales
                if (ema, smoothing) in grid_results
            ]
            smoothing_efficiencies.append(np.mean(efficiencies) if efficiencies else 0)

        ax4.plot(progress_smoothings, smoothing_efficiencies, marker="s", linewidth=2)
        ax4.set_xlabel("Progress Smoothing")
        ax4.set_ylabel("Average Learning Efficiency")
        ax4.set_title("Progress Smoothing Sensitivity")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save locally
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Grid search visualization saved to: {output_path}")

        # Log to WandB
        DynamicalCurriculumVisualizer._log_plot_to_wandb(fig, "grid_search_analysis", log_to_wandb)

        plt.close()

    @staticmethod
    def create_performance_comparison_plot(
        results: Dict[str, Dict], output_path: str = "performance_comparison.png", log_to_wandb: bool = True
    ):
        """Create a performance comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Learning efficiency comparison
        curricula = []
        efficiencies = []
        for curriculum_name, result in results.items():
            if curriculum_name != "grid_search":
                curricula.append(curriculum_name.replace("_", " ").title())
                efficiencies.append(result["learning_efficiency"])

        ax1.bar(curricula, efficiencies, alpha=0.8)
        ax1.set_ylabel("Learning Efficiency")
        ax1.set_title("Learning Efficiency Comparison")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Final performance comparison
        final_performances = []
        for curriculum_name, result in results.items():
            if curriculum_name != "grid_search":
                final_performances.append(result["final_performance"])

        ax2.bar(curricula, final_performances, alpha=0.8)
        ax2.set_ylabel("Final Performance")
        ax2.set_title("Final Performance Comparison")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save locally
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Performance comparison plot saved to: {output_path}")

        # Log to WandB
        DynamicalCurriculumVisualizer._log_plot_to_wandb(fig, "performance_comparison", log_to_wandb)

        plt.close()

    @staticmethod
    def create_sampling_analysis_plot(
        results: Dict[str, Dict], output_path: str = "sampling_analysis.png", log_to_wandb: bool = True
    ):
        """Create a sampling pattern analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Sampling pattern heatmap
        DynamicalCurriculumVisualizer._plot_sampling_patterns(results, ax1)

        # Plot 2: Performance variance over time
        DynamicalCurriculumVisualizer._plot_performance_variance(results, ax2)

        plt.tight_layout()

        # Save locally
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Sampling analysis plot saved to: {output_path}")

        # Log to WandB
        DynamicalCurriculumVisualizer._log_plot_to_wandb(fig, "sampling_analysis", log_to_wandb)

        plt.close()


def create_animation_from_results(
    results: Dict[str, Dict], output_path: str = "curriculum_animation.gif", fps: int = 10, log_to_wandb: bool = True
):
    """Create an animation showing performance evolution over time."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Find the maximum number of epochs
    max_frames = 0
    for curriculum_name, result in results.items():
        if curriculum_name != "grid_search":
            max_frames = max(max_frames, len(result["performance_history"]))

    if max_frames == 0:
        print("No performance data found for animation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(frame):
        ax.clear()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, (curriculum_name, result) in enumerate(results.items()):
            if curriculum_name == "grid_search":
                continue

            performance_history = result["performance_history"]
            if frame < len(performance_history):
                epochs = range(frame + 1)
                performance = performance_history[: frame + 1]

                ax.plot(
                    epochs,
                    performance,
                    label=curriculum_name.replace("_", " ").title(),
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Average Performance")
        ax.set_title(f"Performance Evolution (Epoch {frame})")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, animate, frames=max_frames, interval=1000 / fps, repeat=True)

    # Save locally
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Animation saved to: {output_path}")

    # Log to WandB if available
    if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"performance_animation": wandb.Video(output_path, fps=fps, format="gif")})
        print("Animation logged to WandB")

    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Dynamical Curriculum Visualization Module")
    print("Use this module with the results from dynamical_curriculum_analysis.py")
