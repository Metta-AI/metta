#!/usr/bin/env python3
"""
Dynamical Curriculum Analysis Demo using Existing Codebase

This script demonstrates curriculum learning analysis using the existing
curriculum implementations from the main branch, integrated with a simple
dynamical environment for testing.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metta.rl.dynamical_curriculum_analysis import (
    DynamicalCurriculumAnalysis,
    GridSearchConfig,
    LearningDynamicsConfig,
    create_chain_dependency_graph,
    run_dynamical_analysis,
)
from metta.rl.dynamical_curriculum_visualization import DynamicalCurriculumVisualizer, create_animation_from_results

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to initialize WandB
try:
    import wandb

    WANDB_AVAILABLE = True
    # Initialize WandB run
    wandb.init(
        project="dynamical-curriculum-analysis",
        name="focused-learning-demo",
        config={"analysis_type": "dynamical_curriculum_learning", "version": "1.0"},
    )
    print("WandB initialized successfully")
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Visualizations will only be saved locally.")
except Exception as e:
    WANDB_AVAILABLE = False
    print(f"Failed to initialize WandB: {e}. Visualizations will only be saved locally.")


def run_basic_demo():
    """Run a basic demonstration of the dynamical curriculum analysis."""
    print("=" * 80)
    print("DYNAMICAL CURRICULUM LEARNING ANALYSIS DEMO (OSCILLATORY VERSION)")
    print("=" * 80)
    print("\nThis demo uses existing curriculum implementations from the main branch:")
    print("• RandomCurriculum - Uniform sampling")
    print("• LearningProgressCurriculum - EMA-based adaptive sampling (OSCILLATORY)")
    print("• PrioritizeRegressedCurriculum - Focus on regressed tasks")
    print("• OracleCurriculum - Simple focus strategy")
    print("• Graph-based dynamical environment with transfer learning")
    print("\n" + "=" * 80)

    # Use oscillatory LP parameters
    lp_ema_timescale = 0.01  # 10x larger for more responsive EMAs
    lp_progress_smoothing = 0.01  # 5x smaller for more differentiation

    # Run the analysis with chain graph and oscillatory LP
    results = run_dynamical_analysis(
        num_tasks=10,
        num_epochs=200,
        config=LearningDynamicsConfig(),
        graph_type="chain",
        lp_ema_timescale=lp_ema_timescale,
        lp_progress_smoothing=lp_progress_smoothing,
    )

    # Create visualizations
    print("\nCreating visualizations...")

    # Comprehensive visualization
    DynamicalCurriculumVisualizer.create_comprehensive_visualization(
        results, output_path="dynamical_curriculum_analysis.png", log_to_wandb=WANDB_AVAILABLE
    )

    # Performance comparison plot
    DynamicalCurriculumVisualizer.create_performance_comparison_plot(
        results, output_path="performance_comparison.png", log_to_wandb=WANDB_AVAILABLE
    )

    # Sampling analysis plot
    DynamicalCurriculumVisualizer.create_sampling_analysis_plot(
        results, output_path="sampling_analysis.png", log_to_wandb=WANDB_AVAILABLE
    )

    # Try to create animation
    try:
        create_animation_from_results(
            results, output_path="curriculum_animation.gif", fps=5, log_to_wandb=WANDB_AVAILABLE
        )
    except Exception as e:
        print(f"Animation creation failed: {e}")

    # Log metrics to WandB
    log_metrics_to_wandb(results, "basic_demo")

    return results


def log_metrics_to_wandb(results: Dict[str, Dict], config_name: str = "default"):
    """Log metrics to WandB if available."""
    if not WANDB_AVAILABLE:
        return

    try:
        # Log curriculum metrics
        for curriculum_name, metrics in results.items():
            if curriculum_name != "grid_search":
                wandb.log(
                    {
                        f"{config_name}/{curriculum_name}/learning_efficiency": metrics["learning_efficiency"],
                        f"{config_name}/{curriculum_name}/final_performance": metrics["final_performance"],
                        f"{config_name}/{curriculum_name}/time_to_threshold": metrics["time_to_threshold"] or 0,
                    }
                )

        # Log grid search results if available
        if "grid_search" in results:
            grid_results = results["grid_search"]
            best_params = max(grid_results.items(), key=lambda x: x[1]["learning_efficiency"])
            wandb.log(
                {
                    f"{config_name}/grid_search/best_ema_timescale": best_params[0][0],
                    f"{config_name}/grid_search/best_progress_smoothing": best_params[0][1],
                    f"{config_name}/grid_search/best_efficiency": best_params[1]["learning_efficiency"],
                    f"{config_name}/grid_search/total_combinations": len(grid_results),
                }
            )

        print(f"Metrics logged to WandB for {config_name}")
    except Exception as e:
        print(f"Failed to log metrics to WandB: {e}")


def run_focused_learning_demo():
    """Run a demonstration of the focused learning incentives."""
    print("\n" + "=" * 80)
    print("FOCUSED LEARNING INCENTIVES DEMO (OSCILLATORY VERSION)")
    print("=" * 80)

    # Test different focused learning configurations with more oscillatory LP parameters
    configs = {
        "Baseline (No Focus)": LearningDynamicsConfig(),
        "Focus Bonus": LearningDynamicsConfig(focus_bonus=0.5, focus_decay=0.05),
        "Specialization Bonus": LearningDynamicsConfig(specialization_bonus=0.3, specialization_threshold=0.6),
        "Exploration Penalty": LearningDynamicsConfig(exploration_penalty=0.1),
        "Reduced Dependencies": LearningDynamicsConfig(dependency_strength=0.3),
        "Combined Focus": LearningDynamicsConfig(
            focus_bonus=0.3, specialization_bonus=0.2, exploration_penalty=0.05, dependency_strength=0.5
        ),
        "Oscillatory LP": LearningDynamicsConfig(
            focus_bonus=0.4, specialization_bonus=0.3, exploration_penalty=0.02, dependency_strength=0.4
        ),
    }

    focused_results = {}

    for config_name, config in configs.items():
        print(f"\nTesting: {config_name}")
        print(f"  Focus bonus: {config.focus_bonus}")
        print(f"  Specialization bonus: {config.specialization_bonus}")
        print(f"  Exploration penalty: {config.exploration_penalty}")
        print(f"  Dependency strength: {config.dependency_strength}")

        # Use more oscillatory LP parameters for all configs
        lp_ema_timescale = 0.01  # 10x larger for more responsive EMAs
        lp_progress_smoothing = 0.01  # 5x smaller for more differentiation

        # Run analysis with this configuration and oscillatory LP
        result = run_dynamical_analysis(
            num_tasks=6,
            num_epochs=120,
            config=config,
            graph_type="chain",
            lp_ema_timescale=lp_ema_timescale,
            lp_progress_smoothing=lp_progress_smoothing,
        )

        focused_results[config_name] = result

        # Print key metrics
        print("  Results:")
        for curriculum, metrics in result.items():
            if curriculum != "grid_search":
                print(
                    f"    {curriculum}: efficiency={metrics['learning_efficiency']:.2f}, "
                    f"final_perf={metrics['final_performance']:.3f}"
                )

        # Log metrics to WandB
        log_metrics_to_wandb(result, config_name)

    # Create focused learning comparison visualization
    create_focused_learning_comparison(focused_results)

    return focused_results


def create_focused_learning_comparison(focused_results: Dict[str, Dict]):
    """Create a comparison visualization for focused learning configurations."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Oracle performance across configurations
    ax1 = axes[0, 0]
    configs = list(focused_results.keys())
    oracle_efficiencies = []
    oracle_performances = []

    for config in configs:
        results = focused_results[config]
        if "oracle" in results:
            oracle_efficiencies.append(results["oracle"]["learning_efficiency"])
            oracle_performances.append(results["oracle"]["final_performance"])
        else:
            oracle_efficiencies.append(0)
            oracle_performances.append(0)

    x = np.arange(len(configs))
    width = 0.35

    ax1.bar(x - width / 2, oracle_efficiencies, width, label="Learning Efficiency", alpha=0.8)
    ax1.bar(x + width / 2, [p * 100 for p in oracle_performances], width, label="Final Performance (%)", alpha=0.8)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Performance")
    ax1.set_title("Oracle Performance by Configuration")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Curriculum comparison for best configuration
    ax2 = axes[0, 1]
    best_config = "Combined Focus"  # Based on previous results
    if best_config in focused_results:
        results = focused_results[best_config]
        curricula = ["oracle", "learning_progress", "prioritize_regressed", "random"]
        efficiencies = []

        for curriculum in curricula:
            if curriculum in results:
                efficiencies.append(results[curriculum]["learning_efficiency"])
            else:
                efficiencies.append(0)

        ax2.bar(curricula, efficiencies, alpha=0.8)
        ax2.set_xlabel("Curriculum")
        ax2.set_ylabel("Learning Efficiency")
        ax2.set_title(f"Curriculum Performance ({best_config})")
        ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Focus bonus effect
    ax3 = axes[1, 0]
    focus_configs = ["Baseline (No Focus)", "Focus Bonus", "Combined Focus"]
    focus_efficiencies = []

    for config in focus_configs:
        if config in focused_results:
            results = focused_results[config]
            if "oracle" in results:
                focus_efficiencies.append(results["oracle"]["learning_efficiency"])
            else:
                focus_efficiencies.append(0)
        else:
            focus_efficiencies.append(0)

    ax3.plot(focus_configs, focus_efficiencies, marker="o", linewidth=2, alpha=0.8)
    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("Oracle Learning Efficiency")
    ax3.set_title("Effect of Focus Incentives on Oracle")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Dependency strength effect
    ax4 = axes[1, 1]
    dep_configs = ["Baseline (No Focus)", "Reduced Dependencies", "Combined Focus"]
    dep_efficiencies = []

    for config in dep_configs:
        if config in focused_results:
            results = focused_results[config]
            if "oracle" in results:
                dep_efficiencies.append(results["oracle"]["learning_efficiency"])
            else:
                dep_efficiencies.append(0)
        else:
            dep_efficiencies.append(0)

    ax4.plot(dep_configs, dep_efficiencies, marker="s", linewidth=2, alpha=0.8)
    ax4.set_xlabel("Configuration")
    ax4.set_ylabel("Oracle Learning Efficiency")
    ax4.set_title("Effect of Dependency Strength on Oracle")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("focused_learning_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Focused learning comparison saved to: focused_learning_comparison.png")


def run_graph_comparison_demo():
    """Run a comparison between different graph structures."""
    print("\n" + "=" * 80)
    print("GRAPH STRUCTURE COMPARISON")
    print("=" * 80)

    # Test different graph types
    graph_types = ["chain", "binary_tree"]
    graph_results = {}

    for graph_type in graph_types:
        print(f"\nTesting {graph_type} graph structure...")
        results = run_dynamical_analysis(
            num_tasks=7,  # Use 7 tasks for binary tree (3 levels)
            num_epochs=150,
            config=LearningDynamicsConfig(),
            graph_type=graph_type,
        )
        graph_results[graph_type] = results

    # Create comparison visualization
    create_graph_comparison_visualization(graph_results)

    return graph_results


def create_graph_comparison_visualization(graph_results: Dict[str, Dict]):
    """Create a comparison visualization for different graph structures."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Learning efficiency comparison
    ax1 = axes[0, 0]
    graph_types = list(graph_results.keys())
    curricula = ["oracle", "learning_progress", "random", "prioritize_regressed"]

    for _i, curriculum in enumerate(curricula):
        efficiencies = []
        for graph_type in graph_types:
            if curriculum in graph_results[graph_type]:
                efficiencies.append(graph_results[graph_type][curriculum]["learning_efficiency"])
            else:
                efficiencies.append(0)

        ax1.plot(
            graph_types, efficiencies, marker="o", label=curriculum.replace("_", " ").title(), linewidth=2, alpha=0.8
        )

    ax1.set_xlabel("Graph Structure")
    ax1.set_ylabel("Learning Efficiency")
    ax1.set_title("Learning Efficiency by Graph Structure")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time to threshold comparison
    ax2 = axes[0, 1]
    for _i, curriculum in enumerate(curricula):
        times = []
        for graph_type in graph_types:
            if curriculum in graph_results[graph_type]:
                time_to_thresh = graph_results[graph_type][curriculum]["time_to_threshold"]
                times.append(time_to_thresh if time_to_thresh is not None else 200)
            else:
                times.append(200)

        ax2.plot(graph_types, times, marker="s", label=curriculum.replace("_", " ").title(), linewidth=2, alpha=0.8)

    ax2.set_xlabel("Graph Structure")
    ax2.set_ylabel("Time to Threshold")
    ax2.set_title("Time to Success by Graph Structure")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Performance trajectories for chain graph
    ax3 = axes[1, 0]
    chain_results = graph_results["chain"]

    for curriculum_name, result in chain_results.items():
        performance_history = result["performance_history"]
        epochs = range(len(performance_history))
        ax3.plot(epochs, performance_history, label=curriculum_name.replace("_", " ").title(), linewidth=2, alpha=0.8)

    ax3.set_xlabel("Training Epoch")
    ax3.set_ylabel("Average Performance")
    ax3.set_title("Performance Trajectories (Chain Graph)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance trajectories for binary tree graph
    ax4 = axes[1, 1]
    binary_results = graph_results["binary_tree"]

    for curriculum_name, result in binary_results.items():
        performance_history = result["performance_history"]
        epochs = range(len(performance_history))
        ax4.plot(epochs, performance_history, label=curriculum_name.replace("_", " ").title(), linewidth=2, alpha=0.8)

    ax4.set_xlabel("Training Epoch")
    ax4.set_ylabel("Average Performance")
    ax4.set_title("Performance Trajectories (Binary Tree Graph)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Graph comparison saved to: graph_comparison.png")


def run_grid_search_demo():
    """Run a demonstration of grid search over learning progress parameters."""
    print("\n" + "=" * 80)
    print("GRID SEARCH: LEARNING PROGRESS PARAMETER OPTIMIZATION")
    print("=" * 80)

    # Define grid search configuration
    grid_config = GridSearchConfig(
        ema_timescales=[0.00001, 0.0001, 0.001, 0.01, 0.1],
        progress_smoothings=[0.001, 0.01, 0.05, 0.1, 0.2],
        num_epochs=100,
    )

    print("\nGrid search configuration:")
    print(f"  EMA timescales: {grid_config.ema_timescales}")
    print(f"  Progress smoothings: {grid_config.progress_smoothings}")
    print(f"  Total combinations: {len(grid_config.ema_timescales) * len(grid_config.progress_smoothings)}")
    print(f"  Epochs per run: {grid_config.num_epochs}")

    # Run the analysis with grid search
    results = run_dynamical_analysis(
        num_tasks=8, num_epochs=100, config=LearningDynamicsConfig(), run_grid_search=True, grid_config=grid_config
    )

    # Create grid search visualizations
    print("\nCreating grid search visualizations...")

    if "grid_search" in results:
        # Comprehensive grid search visualization
        DynamicalCurriculumVisualizer.create_grid_search_visualization(
            results["grid_search"], output_path="grid_search_analysis.png", log_to_wandb=WANDB_AVAILABLE
        )

        # Update comprehensive visualization to include grid search
        DynamicalCurriculumVisualizer.create_comprehensive_visualization(
            results, output_path="dynamical_curriculum_analysis_with_grid_search.png", log_to_wandb=WANDB_AVAILABLE
        )

    return results


def run_parameter_sweep():
    """Run a parameter sweep to explore different learning dynamics configurations."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP: EXPLORING LEARNING DYNAMICS")
    print("=" * 80)

    # Define parameter configurations to test
    configs = [
        LearningDynamicsConfig(gamma=0.1, lambda_forget=0.005),
        LearningDynamicsConfig(gamma=0.3, lambda_forget=0.01),
        LearningDynamicsConfig(gamma=0.5, lambda_forget=0.02),
    ]

    config_names = ["Low Transfer", "Medium Transfer", "High Transfer"]

    all_results = {}

    for config, name in zip(configs, config_names, strict=False):
        print(f"\nTesting configuration: {name}")
        print(f"  Gamma: {config.gamma}")
        print(f"  Lambda: {config.lambda_forget}")

        results = run_dynamical_analysis(num_tasks=8, num_epochs=150, config=config)

        all_results[name] = results

    # Create comparison visualization
    print("\nCreating parameter sweep comparison...")
    create_parameter_sweep_comparison(all_results)

    return all_results


def create_parameter_sweep_comparison(all_results: Dict[str, Dict]):
    """Create a comparison visualization for the parameter sweep."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Learning efficiency comparison
    ax1 = axes[0, 0]
    configs = list(all_results.keys())
    curricula = ["oracle", "learning_progress", "random", "prioritize_regressed"]

    for i, curriculum in enumerate(curricula):
        efficiencies = []
        for config in configs:
            if curriculum in all_results[config]:
                efficiencies.append(all_results[config][curriculum]["learning_efficiency"])
            else:
                efficiencies.append(0)

        x = np.arange(len(configs))
        width = 0.2
        ax1.bar(x + i * width, efficiencies, width, label=curriculum.replace("_", " ").title(), alpha=0.8)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Learning Efficiency")
    ax1.set_title("Learning Efficiency by Configuration")
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(configs)
    ax1.legend()

    # Plot 2: Time to threshold comparison
    ax2 = axes[0, 1]
    for i, curriculum in enumerate(curricula):
        times = []
        for config in configs:
            if curriculum in all_results[config]:
                time_to_thresh = all_results[config][curriculum]["time_to_threshold"]
                times.append(time_to_thresh if time_to_thresh is not None else 200)
            else:
                times.append(200)

        ax2.bar(x + i * width, times, width, label=curriculum.replace("_", " ").title(), alpha=0.8)

    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Time to Threshold")
    ax2.set_title("Time to Success by Configuration")
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels(configs)
    ax2.legend()

    # Plot 3: Performance trajectories for medium transfer
    ax3 = axes[1, 0]
    medium_results = all_results["Medium Transfer"]

    for curriculum_name, result in medium_results.items():
        performance_history = result["performance_history"]
        epochs = range(len(performance_history))
        ax3.plot(epochs, performance_history, label=curriculum_name.replace("_", " ").title(), linewidth=2, alpha=0.8)

    ax3.set_xlabel("Training Epoch")
    ax3.set_ylabel("Average Performance")
    ax3.set_title("Performance Trajectories (Medium Transfer)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final performance comparison
    ax4 = axes[1, 1]
    final_perfs = []
    curricula = ["oracle", "learning_progress", "random", "prioritize_regressed"]

    for config in configs:
        results = all_results[config]
        config_perfs = [results[curriculum]["final_performance"] for curriculum in curricula]
        final_perfs.append(config_perfs)

    final_perfs = np.array(final_perfs).T

    for i, curriculum in enumerate(curricula):
        ax4.plot(
            configs, final_perfs[i], label=curriculum.replace("_", " ").title(), marker="o", linewidth=2, alpha=0.8
        )

    ax4.set_xlabel("Configuration")
    ax4.set_ylabel("Final Performance")
    ax4.set_title("Final Performance by Configuration")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("parameter_sweep_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Parameter sweep comparison saved to: parameter_sweep_comparison.png")


def run_ablation_study():
    """Run an ablation study to understand the impact of different components."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: COMPONENT ANALYSIS")
    print("=" * 80)

    # Create dependency graph
    dependency_graph = create_chain_dependency_graph(6)

    # Test different configurations
    ablation_configs = {
        "Full System": LearningDynamicsConfig(),
        "No Transfer (γ=0)": LearningDynamicsConfig(gamma=0.0),
        "No Forgetting (λ=0)": LearningDynamicsConfig(lambda_forget=0.0),
        "High Transfer": LearningDynamicsConfig(gamma=0.8),
        "High Forgetting": LearningDynamicsConfig(lambda_forget=0.05),
    }

    ablation_results = {}

    for config_name, config in ablation_configs.items():
        print(f"\nTesting: {config_name}")

        analysis = DynamicalCurriculumAnalysis(dependency_graph, config)
        results = analysis.run_curriculum_comparison(100)
        ablation_results[config_name] = results

    # Create ablation visualization
    create_ablation_visualization(ablation_results)

    return ablation_results


def create_ablation_visualization(ablation_results: Dict[str, Dict]):
    """Create visualization for the ablation study."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Learning efficiency comparison
    ax1 = axes[0, 0]
    configs = list(ablation_results.keys())
    oracle_efficiencies = []
    lp_efficiencies = []

    for config in configs:
        results = ablation_results[config]
        oracle_efficiencies.append(results["oracle"]["learning_efficiency"])
        lp_efficiencies.append(results["learning_progress"]["learning_efficiency"])

    x = np.arange(len(configs))
    width = 0.35

    ax1.bar(x - width / 2, oracle_efficiencies, width, label="Oracle", alpha=0.8)
    ax1.bar(x + width / 2, lp_efficiencies, width, label="Learning Progress", alpha=0.8)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Learning Efficiency")
    ax1.set_title("Learning Efficiency by Configuration")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Performance trajectories for full system
    ax2 = axes[0, 1]
    full_results = ablation_results["Full System"]

    for curriculum_name, result in full_results.items():
        performance_history = result["performance_history"]
        epochs = range(len(performance_history))
        ax2.plot(epochs, performance_history, label=curriculum_name.replace("_", " ").title(), linewidth=2, alpha=0.8)

    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("Average Performance")
    ax2.set_title("Performance Trajectories (Full System)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Impact of transfer parameter
    ax3 = axes[1, 0]
    transfer_configs = ["No Transfer (γ=0)", "Full System", "High Transfer"]
    transfer_results = {k: ablation_results[k] for k in transfer_configs}

    for config_name, results in transfer_results.items():
        lp_performance = results["learning_progress"]["performance_history"]
        epochs = range(len(lp_performance))
        ax3.plot(epochs, lp_performance, label=config_name, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Training Epoch")
    ax3.set_ylabel("Learning Progress Performance")
    ax3.set_title("Impact of Transfer Parameter (γ)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Impact of forgetting parameter
    ax4 = axes[1, 1]
    forgetting_configs = ["No Forgetting (λ=0)", "Full System", "High Forgetting"]
    forgetting_results = {k: ablation_results[k] for k in forgetting_configs}

    for config_name, results in forgetting_results.items():
        lp_performance = results["learning_progress"]["performance_history"]
        epochs = range(len(lp_performance))
        ax4.plot(epochs, lp_performance, label=config_name, linewidth=2, alpha=0.8)

    ax4.set_xlabel("Training Epoch")
    ax4.set_ylabel("Learning Progress Performance")
    ax4.set_title("Impact of Forgetting Parameter (λ)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Ablation study saved to: ablation_study.png")


def main():
    """Main function to run the complete dynamical curriculum analysis demo."""
    print("Starting Dynamical Curriculum Analysis Demo (Oscillatory Version)...")

    # Run basic demo
    basic_results = run_basic_demo()

    # Run focused learning demo
    try:
        focused_results = run_focused_learning_demo()
    except Exception as e:
        print(f"Focused learning demo failed: {e}")
        focused_results = {}

    # Run graph comparison demo
    try:
        run_graph_comparison_demo()
    except Exception as e:
        print(f"Graph comparison demo failed: {e}")

    # Run grid search demo
    try:
        grid_search_results = run_grid_search_demo()
    except Exception as e:
        print(f"Grid search demo failed: {e}")
        grid_search_results = {}

    # Run parameter sweep
    try:
        run_parameter_sweep()
    except Exception as e:
        print(f"Parameter sweep failed: {e}")

    # Run ablation study
    try:
        run_ablation_study()
    except Exception as e:
        print(f"Ablation study failed: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("OSCILLATORY DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("• dynamical_curriculum_analysis.png - Basic comprehensive analysis (Oscillatory)")
    print("• focused_learning_comparison.png - Focused learning incentives analysis (Oscillatory)")
    print("• dynamical_curriculum_analysis_with_grid_search.png - Analysis with grid search")
    print("• grid_search_analysis.png - Grid search parameter optimization")
    print("• graph_comparison.png - Graph structure comparison")
    print("• performance_comparison.png - Performance comparison")
    print("• sampling_analysis.png - Sampling pattern analysis")
    print("• curriculum_animation.gif - Performance evolution animation")
    print("• parameter_sweep_comparison.png - Parameter sweep results")
    print("• ablation_study.png - Ablation study results")

    print("\nKey findings (Oscillatory Version):")
    if basic_results:
        oracle_eff = basic_results["oracle"]["learning_efficiency"]
        lp_eff = basic_results["learning_progress"]["learning_efficiency"]
        random_eff = basic_results["random"]["learning_efficiency"]
        pr_eff = basic_results["prioritize_regressed"]["learning_efficiency"]

        print(f"• Oracle efficiency: {oracle_eff:.2f}")
        print(f"• Learning Progress efficiency (Oscillatory): {lp_eff:.2f}")
        print(f"• Prioritize Regressed efficiency: {pr_eff:.2f}")
        print(f"• Random efficiency: {random_eff:.2f}")
        print(f"• LP achieves {lp_eff / oracle_eff * 100:.1f}% of oracle performance")
        print(f"• PR achieves {pr_eff / oracle_eff * 100:.1f}% of oracle performance")
        print(f"• LP is {lp_eff / random_eff:.1f}x more efficient than random")
        print(f"• PR is {pr_eff / random_eff:.1f}x more efficient than random")

    if focused_results and "Combined Focus" in focused_results:
        combined_results = focused_results["Combined Focus"]
        if "oracle" in combined_results:
            focused_oracle_eff = combined_results["oracle"]["learning_efficiency"]
            print(f"• With focused incentives, Oracle efficiency: {focused_oracle_eff:.2f}")
            if basic_results and "oracle" in basic_results:
                improvement = (focused_oracle_eff / basic_results["oracle"]["learning_efficiency"] - 1) * 100
                print(f"• Focused incentives improve Oracle by {improvement:.1f}%")

    if focused_results and "Oscillatory LP" in focused_results:
        oscillatory_results = focused_results["Oscillatory LP"]
        if "learning_progress" in oscillatory_results:
            oscillatory_lp_eff = oscillatory_results["learning_progress"]["learning_efficiency"]
            print(f"• Oscillatory LP efficiency: {oscillatory_lp_eff:.2f}")
            if basic_results and "learning_progress" in basic_results:
                improvement = (oscillatory_lp_eff / basic_results["learning_progress"]["learning_efficiency"] - 1) * 100
                print(f"• Oscillatory LP improves LP by {improvement:.1f}%")

    if grid_search_results and "grid_search" in grid_search_results:
        grid_results = grid_search_results["grid_search"]
        best_params = max(grid_results.items(), key=lambda x: x[1]["learning_efficiency"])
        print("\nGrid Search Results:")
        print(f"• Best EMA timescale: {best_params[0][0]:.6f}")
        print(f"• Best progress smoothing: {best_params[0][1]:.6f}")
        print(f"• Best efficiency: {best_params[1]['learning_efficiency']:.2f}")
        print(f"• Total combinations tested: {len(grid_results)}")

    print("\nThis oscillatory demo demonstrates:")
    print("• Integration with existing curriculum implementations from main branch")
    print("• Graph-based dynamical environment with transfer learning")
    print("• Focused learning incentives (focus bonus, specialization bonus, exploration penalty)")
    print("• OSCILLATORY Learning Progress with aggressive EMA updates (0.01) and low smoothing (0.01)")
    print("• Fair comparison of all curricula using structured task dependencies")
    print("• Grid search optimization of learning progress parameters")
    print("• Comprehensive analysis of learning dynamics and efficiency")
    print("• Parameter sensitivity studies and ablation analysis")
    print("• Comparison of different graph structures (chain vs binary tree)")
    print("• Effect of focused learning incentives on curriculum performance")
    print("• More differentiated and aggressive curriculum behavior")

    # Cleanup WandB
    cleanup_wandb()


def cleanup_wandb():
    """Finish the WandB run if it was initialized."""
    if WANDB_AVAILABLE:
        try:
            wandb.finish()
            print("WandB run finished successfully")
        except Exception as e:
            print(f"Failed to finish WandB run: {e}")


if __name__ == "__main__":
    main()
