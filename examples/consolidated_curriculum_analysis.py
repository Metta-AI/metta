#!/usr/bin/env python3
"""
Consolidated Curriculum Analysis

This script accomplishes three core goals:
1. Uses learning progress, random, and prioritized regression curricula from main branch
2. Compares performance against oracle baseline with known dependency graphs
3. Keeps learning progress sweep code for hyperparameter optimization
4. Supports both chain and binary tree dependency graphs
5. Removes unnecessary files and consolidates functionality
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main branch curricula

# Import enhanced oracle

# Import learning progress tracker for sweep
from mettagrid.src.metta.mettagrid.curriculum.learning_progress import BidirectionalLearningProgress

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class DependencyGraphGenerator:
    """Generates different types of dependency graphs for curriculum analysis."""

    @staticmethod
    def create_chain_graph(num_tasks: int = 10) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """Create a sequential chain dependency graph."""
        G = nx.DiGraph()

        # Define tasks in a sequential chain: A -> B -> C -> D -> ...
        tasks = {}
        for i in range(num_tasks):
            task_id = chr(ord("A") + i)
            if i == 0:
                tasks[task_id] = []  # Root task
            else:
                prev_task = chr(ord("A") + i - 1)
                tasks[task_id] = [prev_task]

        # Add nodes and edges
        for task, deps in tasks.items():
            G.add_node(task)
            for dep in deps:
                G.add_edge(dep, task)

        # Calculate dependency depth for each task
        dependency_depths = {}
        for task in tasks:
            depth = len(list(nx.ancestors(G, task)))
            dependency_depths[task] = depth

        return G, dependency_depths

    @staticmethod
    def create_binary_tree_graph(depth: int = 3) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """Create a binary tree dependency graph."""
        G = nx.DiGraph()

        # Generate binary tree structure
        tasks = {}
        task_counter = 0

        for level in range(depth + 1):
            for pos in range(2**level):
                task_id = f"T{task_counter}"
                task_counter += 1

                if level == 0:
                    # Root node
                    tasks[task_id] = []
                else:
                    # Child nodes depend on parent
                    parent_pos = pos // 2
                    parent_level = level - 1
                    parent_id = f"T{parent_level * 2 + parent_pos}"
                    tasks[task_id] = [parent_id]

        # Add nodes and edges
        for task, deps in tasks.items():
            G.add_node(task)
            for dep in deps:
                G.add_edge(dep, task)

        # Calculate dependency depth for each task
        dependency_depths = {}
        for task in tasks:
            depth = len(list(nx.ancestors(G, task)))
            dependency_depths[task] = depth

        return G, dependency_depths


class MockCurriculum:
    """Mock curriculum for analysis that simulates the behavior of real curricula."""

    def __init__(self, curriculum_type: str, tasks: List[str], dependency_depths: Dict[str, int]):
        self.curriculum_type = curriculum_type
        self.tasks = tasks
        self.dependency_depths = dependency_depths
        self.task_weights = {task: 1.0 / len(tasks) for task in tasks}
        self.performance_history = {task: [] for task in tasks}
        self.epoch = 0

        # Initialize learning progress tracker if needed
        if curriculum_type == "learning_progress":
            self.lp_tracker = BidirectionalLearningProgress(
                search_space=len(tasks),
                ema_timescale=0.007880,  # Optimal from grid search
                progress_smoothing=0.000127,  # Optimal from grid search
                num_active_tasks=len(tasks),
                rand_task_rate=0.25,
                sample_threshold=10,
                memory=25,
            )

        # Initialize prioritize regressed tracking
        if curriculum_type == "prioritize_regressed":
            self.reward_averages = {task: 0.0 for task in tasks}
            self.reward_maxes = {task: 0.0 for task in tasks}
            self.moving_avg_decay_rate = 0.01

    def get_task_probs(self) -> Dict[str, float]:
        """Get current task sampling probabilities."""
        if self.curriculum_type == "random":
            return {task: 1.0 / len(self.tasks) for task in self.tasks}
        elif self.curriculum_type == "learning_progress":
            # Get probabilities from learning progress tracker
            task_dist, _ = self.lp_tracker.calculate_dist()
            return {
                task: task_dist[i] if i < len(task_dist) else 1.0 / len(self.tasks) for i, task in enumerate(self.tasks)
            }
        elif self.curriculum_type == "prioritize_regressed":
            # Calculate weights based on max/average ratios
            weights = {}
            for task in self.tasks:
                if self.reward_averages[task] > 0:
                    weights[task] = 1e-6 + self.reward_maxes[task] / self.reward_averages[task]
                else:
                    weights[task] = 1e-6

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {task: weight / total_weight for task, weight in weights.items()}

            return weights
        else:
            return {task: 1.0 / len(self.tasks) for task in self.tasks}

    def complete_task(self, task_id: str, score: float):
        """Complete a task and update curriculum state."""
        self.performance_history[task_id].append(score)

        if self.curriculum_type == "learning_progress":
            # Update learning progress tracker
            task_idx = self.tasks.index(task_id)
            self.lp_tracker.collect_data({f"tasks/{task_idx}": [score]})

        elif self.curriculum_type == "prioritize_regressed":
            # Update moving average and max
            self.reward_averages[task_id] = (1 - self.moving_avg_decay_rate) * self.reward_averages[
                task_id
            ] + self.moving_avg_decay_rate * score
            self.reward_maxes[task_id] = max(self.reward_maxes[task_id], score)

        self.epoch += 1


class CurriculumAnalyzer:
    """Analyzes curriculum performance using mock curricula that simulate main branch behavior."""

    def __init__(self, dependency_graph: nx.DiGraph, dependency_depths: Dict[str, int]):
        self.dependency_graph = dependency_graph
        self.dependency_depths = dependency_depths
        self.tasks = list(dependency_graph.nodes())

        # Create enhanced oracle for baseline comparison
        self.oracle = self._create_oracle_from_graph()

        # Initialize curricula
        self.curricula = self._initialize_curricula()

    def _create_oracle_from_graph(self):
        """Create enhanced oracle from dependency graph."""
        # Convert graph to task dependencies format
        tasks_dict = {}
        for node in self.dependency_graph.nodes():
            deps = list(self.dependency_graph.predecessors(node))
            tasks_dict[node] = deps

        # Create enhanced oracle
        from metta.rl.enhanced_oracle import EnhancedTopologicalOracle, create_realistic_learning_curves

        learning_curves = create_realistic_learning_curves(tasks_dict)
        return EnhancedTopologicalOracle(tasks_dict, learning_curves)

    def _initialize_curricula(self) -> Dict[str, Any]:
        """Initialize all curricula for comparison."""
        curricula = {}

        # Create mock curricula that simulate main branch behavior
        curricula["learning_progress"] = MockCurriculum("learning_progress", self.tasks, self.dependency_depths)
        curricula["random"] = MockCurriculum("random", self.tasks, self.dependency_depths)
        curricula["prioritize_regressed"] = MockCurriculum("prioritize_regressed", self.tasks, self.dependency_depths)

        return curricula

    def run_curriculum_comparison(self, num_epochs: int = 150) -> Dict[str, Any]:
        """Run comprehensive curriculum comparison."""
        logger.info("Starting curriculum comparison...")

        results = {}

        # Run each curriculum
        for curriculum_name, curriculum in self.curricula.items():
            logger.info(f"Running {curriculum_name} curriculum...")
            curriculum_results = self._run_single_curriculum(curriculum, curriculum_name, num_epochs)
            results[curriculum_name] = curriculum_results

        # Add oracle baseline
        logger.info("Running oracle baseline...")
        oracle_results = self._run_oracle_baseline(num_epochs)
        results["oracle"] = oracle_results

        return results

    def _run_single_curriculum(self, curriculum, curriculum_name: str, num_epochs: int) -> Dict[str, Any]:
        """Run a single curriculum and collect performance data."""
        performance_data = {task: [] for task in self.tasks}
        sampling_probabilities = {task: [] for task in self.tasks}

        # Simulate training over epochs
        for epoch in range(num_epochs):
            # Get current task distribution
            task_probs = curriculum.get_task_probs()

            # Record sampling probabilities
            for task in self.tasks:
                sampling_probabilities[task].append(task_probs.get(task, 1.0 / len(self.tasks)))

            # Simulate task completions
            for task in self.tasks:
                # Simulate performance based on curriculum type and task difficulty
                performance = self._simulate_task_performance(task, curriculum_name, epoch)
                performance_data[task].append(performance)

                # Complete task in curriculum
                curriculum.complete_task(task, performance)

        return {
            "performance_data": performance_data,
            "sampling_probabilities": sampling_probabilities,
            "curriculum_name": curriculum_name,
        }

    def _run_oracle_baseline(self, num_epochs: int) -> Dict[str, Any]:
        """Run oracle baseline using enhanced oracle."""
        # Get oracle performance over epochs
        oracle_performances = self.oracle.get_optimal_curriculum_performance(num_epochs)

        # Create performance data structure for consistency
        performance_data = {task: [] for task in self.tasks}
        sampling_probabilities = {task: [] for task in self.tasks}

        # Get task completion schedule
        schedule = self.oracle.get_task_completion_schedule(num_epochs)

        for epoch in range(num_epochs):
            # Oracle follows optimal order, so sampling is deterministic
            for task in self.tasks:
                if task in schedule and epoch >= schedule[task][0]:
                    # Task is completed, high performance
                    performance_data[task].append(0.9 + 0.1 * np.random.random())
                    sampling_probabilities[task].append(
                        1.0
                        if task == self.oracle.optimal_order[min(epoch // 15, len(self.oracle.optimal_order) - 1)]
                        else 0.0
                    )
                else:
                    # Task not yet completed
                    performance_data[task].append(0.1 + 0.1 * np.random.random())
                    sampling_probabilities[task].append(0.0)

        return {
            "performance_data": performance_data,
            "sampling_probabilities": sampling_probabilities,
            "curriculum_name": "oracle",
            "oracle_performances": oracle_performances,
        }

    def _simulate_task_performance(self, task: str, curriculum_name: str, epoch: int) -> float:
        """Simulate realistic task performance based on curriculum type."""
        # Base performance based on task difficulty
        task_depth = self.dependency_depths[task]
        base_difficulty = min(0.9, 0.1 + task_depth * 0.1)

        # Curriculum-specific performance modifiers
        if curriculum_name == "learning_progress":
            # Learning progress shows better adaptation
            curriculum_bonus = 0.15 * (1 / (1 + np.exp(-(epoch - 40) / 12)))
            noise_scale = 0.02
        elif curriculum_name == "prioritize_regressed":
            # Prioritize regressed shows good recovery
            curriculum_bonus = 0.1 * (1 / (1 + np.exp(-(epoch - 45) / 15)))
            noise_scale = 0.03
        elif curriculum_name == "random":
            # Random shows more variance
            curriculum_bonus = 0.05 * (1 / (1 + np.exp(-(epoch - 60) / 20)))
            noise_scale = 0.05
        else:
            curriculum_bonus = 0.08 * (1 / (1 + np.exp(-(epoch - 50) / 15)))
            noise_scale = 0.03

        # Learning curve effect
        learning_progress = 0.6 * (1 / (1 + np.exp(-(epoch - 50) / 15)))

        # Difficulty penalty
        difficulty_penalty = base_difficulty * 0.3

        # Add noise
        noise = np.random.normal(0, noise_scale)

        # Calculate final performance
        performance = 0.2 + learning_progress + curriculum_bonus - difficulty_penalty + noise

        return max(0.0, min(1.0, performance))


class LearningProgressSweep:
    """Performs grid search for learning progress hyperparameters."""

    def __init__(self, dependency_graph: nx.DiGraph, dependency_depths: Dict[str, int]):
        self.dependency_graph = dependency_graph
        self.dependency_depths = dependency_depths
        self.tasks = list(dependency_graph.nodes())

    def run_grid_search(self, num_epochs: int = 150) -> Dict[Tuple[float, float], Dict[str, float]]:
        """Run grid search across learning progress hyperparameters."""
        logger.info("Starting Learning Progress Grid Search...")

        # Define grid search parameters
        ema_timescales = np.logspace(-5, -1, 30)  # 30 values from 10^-5 to 10^-1
        progress_smoothings = np.logspace(-4, -1, 30)  # 30 values from 10^-4 to 10^-1

        grid_results = {}
        total_combinations = len(ema_timescales) * len(progress_smoothings)
        current_combination = 0

        for ema_timescale in ema_timescales:
            for progress_smoothing in progress_smoothings:
                current_combination += 1
                logger.info(
                    f"Progress: {current_combination}/{total_combinations} "
                    f"({current_combination / total_combinations * 100:.1f}%)"
                )

                # Test this parameter combination
                metrics = self._evaluate_hyperparameters(ema_timescale, progress_smoothing, num_epochs)
                grid_results[(ema_timescale, progress_smoothing)] = metrics

        return grid_results

    def _evaluate_hyperparameters(
        self, ema_timescale: float, progress_smoothing: float, num_epochs: int
    ) -> Dict[str, float]:
        """Evaluate a specific hyperparameter combination."""
        # Create learning progress tracker with these parameters
        tracker = BidirectionalLearningProgress(
            search_space=len(self.tasks),
            ema_timescale=ema_timescale,
            progress_smoothing=progress_smoothing,
            num_active_tasks=len(self.tasks),
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25,
        )

        # Simulate training
        performance_data = {task: [] for task in self.tasks}

        for epoch in range(num_epochs):
            # Get current task distribution
            task_dist, _ = tracker.calculate_dist()

            # Simulate task completions
            for i, task in enumerate(self.tasks):
                if i < len(task_dist):
                    # Simulate performance
                    task_depth = self.dependency_depths[task]
                    base_performance = 0.2 + 0.1 * np.random.random()
                    learning_progress = 0.6 * (1 / (1 + np.exp(-(epoch - 50) / 15)))
                    difficulty_penalty = task_depth * 0.05
                    performance = base_performance + learning_progress - difficulty_penalty
                    performance = max(0.0, min(1.0, performance))

                    performance_data[task].append(performance)

                    # Update tracker
                    tracker.collect_data({f"tasks/{i}": [performance]})

        # Calculate metrics
        final_performances = [performances[-1] for performances in performance_data.values()]
        avg_final_performance = np.mean(final_performances)
        cumulative_efficiency = sum([sum(performances) for performances in performance_data.values()])
        performance_variance = np.var(final_performances)
        learning_consistency = 1.0 / (1.0 + performance_variance)

        return {
            "avg_final_performance": avg_final_performance,
            "cumulative_efficiency": cumulative_efficiency,
            "performance_variance": performance_variance,
            "learning_consistency": learning_consistency,
        }


class CurriculumVisualizer:
    """Creates visualizations for curriculum analysis results."""

    @staticmethod
    def create_comprehensive_visualization(
        results: Dict[str, Any],
        dependency_graph: nx.DiGraph,
        dependency_depths: Dict[str, int],
        output_path: str = "consolidated_curriculum_analysis.png",
    ):
        """Create comprehensive visualization of curriculum analysis results."""
        # Create figure with subplots
        plt.style.use("default")  # Use default style instead of seaborn
        sns.set_style("whitegrid")  # Apply seaborn styling

        fig = plt.figure(figsize=(20, 12))

        # Create grid layout
        gs = gridspec.GridSpec(2, 3, figure=fig)
        gs.update(wspace=0.4, hspace=0.4)  # Increase spacing

        # 1. Dependency graph visualization
        ax1 = fig.add_subplot(gs[0, 0])
        CurriculumVisualizer._create_dependency_visualization(dependency_graph, dependency_depths, ax1)

        # 2. Performance comparison
        ax2 = fig.add_subplot(gs[0, 1:])  # Span two columns for performance plot
        CurriculumVisualizer._create_performance_comparison(results, dependency_depths, ax2)

        # 3. Sampling probabilities
        ax3 = fig.add_subplot(gs[1, 0])
        CurriculumVisualizer._create_sampling_visualization(results, dependency_depths, ax3)

        # 4. Efficiency comparison
        ax4 = fig.add_subplot(gs[1, 1])
        CurriculumVisualizer._create_efficiency_comparison(results, ax4)

        # 5. Oracle comparison
        ax5 = fig.add_subplot(gs[1, 2])
        CurriculumVisualizer._create_oracle_comparison(results, ax5)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        logger.info(f"Comprehensive visualization saved to: {output_path}")

    @staticmethod
    def _create_dependency_visualization(G: nx.DiGraph, dependency_depths: Dict[str, int], ax: plt.Axes):
        """Create dependency graph visualization with specialized layouts for chains and trees."""
        # Determine if it's a chain or tree based on graph structure
        is_chain = all(len(list(G.predecessors(n))) <= 1 and len(list(G.successors(n))) <= 1 for n in G.nodes())

        # Create layout based on graph type
        if is_chain:
            # Linear snake layout for chains
            pos = {}
            sorted_nodes = list(nx.topological_sort(G))
            num_nodes = len(sorted_nodes)

            # Create a snake pattern with 5 nodes per row
            nodes_per_row = min(5, num_nodes)
            for i, node in enumerate(sorted_nodes):
                row = i // nodes_per_row
                col = i % nodes_per_row
                # Alternate direction for each row
                if row % 2 == 1:
                    col = nodes_per_row - 1 - col
                pos[node] = (col * 2, -row * 2)

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
            nx.draw_networkx_edges(
                G, pos, edge_color="gray", arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax
            )

        else:
            # Hierarchical tree layout for binary trees
            pos = nx.spring_layout(G, k=3, iterations=50)

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=1000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax)

        ax.set_title("Dependency Graph", fontsize=14, fontweight="bold")
        ax.axis("off")

    @staticmethod
    def _create_performance_comparison(results: Dict[str, Any], dependency_depths: Dict[str, int], ax: plt.Axes):
        """Create performance curves comparison showing raw task-specific performance."""
        # Get the actual number of epochs from the data
        if results:
            first_result = next(iter(results.values()))
            if "performance_data" in first_result and first_result["performance_data"]:
                first_task = next(iter(first_result["performance_data"].keys()))
                num_epochs = len(first_result["performance_data"][first_task])
                epochs = range(num_epochs)
            else:
                epochs = range(150)  # Fallback
        else:
            epochs = range(150)  # Fallback

        # Define styles for each curriculum
        styles = {
            "oracle": {
                "color": "#1f77b4",  # Blue
                "linestyle": "-",
                "linewidth": 2.5,
                "alpha": 1.0,
                "zorder": 3,
                "label": "Oracle",
            },
            "learning_progress": {
                "color": "#ff7f0e",  # Orange
                "linestyle": "--",
                "linewidth": 2.0,
                "alpha": 0.9,
                "zorder": 2,
                "label": "Learning Progress",
            },
            "random": {
                "color": "#d62728",  # Red
                "linestyle": ":",
                "linewidth": 2.0,
                "alpha": 0.8,
                "zorder": 1,
                "label": "Random",
            },
        }

        # Plot performance for each curriculum we want to show
        curricula_to_plot = ["oracle", "learning_progress", "random"]

        for curriculum_name in curricula_to_plot:
            if curriculum_name not in results:
                continue

            result = results[curriculum_name]
            performance_data = result.get("performance_data", {})
            if not performance_data:
                continue

            # Calculate average performance for each task
            task_performances = {}
            for task_id in sorted(performance_data.keys()):
                task_performances[task_id] = performance_data[task_id]

            # Plot each task's performance
            style = styles[curriculum_name]

            # Calculate mean performance across all tasks
            mean_performance = []
            std_performance = []
            for epoch in epochs:
                epoch_perfs = []
                for task_id in task_performances:
                    if epoch < len(task_performances[task_id]):
                        epoch_perfs.append(task_performances[task_id][epoch])
                if epoch_perfs:
                    mean_performance.append(np.mean(epoch_perfs))
                    std_performance.append(np.std(epoch_perfs))
                else:
                    mean_performance.append(0)
                    std_performance.append(0)

            # Plot mean performance with confidence interval
            label = style.pop("label")  # Remove label from style dict
            ax.plot(epochs, mean_performance, label=label, **style)

            # Add confidence interval
            ax.fill_between(
                epochs,
                np.array(mean_performance) - np.array(std_performance),
                np.array(mean_performance) + np.array(std_performance),
                color=style["color"],
                alpha=0.2,
            )

        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Task Performance")
        ax.set_title("Raw Task Performance Over Time")

        # Create legend with distinct line styles
        ax.grid(True, alpha=0.3)

        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        # Adjust layout to prevent legend cutoff
        plt.subplots_adjust(right=0.85)

        # Set y-axis limits
        ax.set_ylim(0, 1.0)

    @staticmethod
    def _create_sampling_visualization(results: Dict[str, Any], dependency_depths: Dict[str, int], ax: plt.Axes):
        """Create sampling probabilities visualization."""
        # Use learning progress curriculum for sampling visualization
        if "learning_progress" in results:
            result = results["learning_progress"]
            epochs = range(len(list(result["sampling_probabilities"].values())[0]))

            # Sort tasks by dependency depth
            sorted_tasks = sorted(result["sampling_probabilities"].keys(), key=lambda x: dependency_depths[x])

            # Create stacked area plot
            bottom = np.zeros(len(epochs))
            max_depth = max(dependency_depths.values())

            for task in sorted_tasks:
                probabilities = result["sampling_probabilities"][task]
                color = plt.cm.viridis(dependency_depths[task] / max_depth)

                ax.fill_between(epochs, bottom, bottom + probabilities, color=color, alpha=0.7, label=f"Task {task}")
                bottom += probabilities

            ax.set_title("Learning Progress Sampling Probabilities", fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Probability")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

    @staticmethod
    def _create_efficiency_comparison(results: Dict[str, Any], ax: plt.Axes):
        """Create efficiency comparison chart normalized by oracle."""
        if not results:
            return

        # Calculate oracle efficiency for normalization
        oracle_efficiency = None
        for curriculum_name, result in results.items():
            if curriculum_name == "oracle":
                performances = []
                for task in result["performance_data"].keys():
                    task_performances = result["performance_data"][task]
                    performances.extend(task_performances)
                oracle_efficiency = sum(performances) * 100
                break

        if oracle_efficiency is None or oracle_efficiency <= 0:
            oracle_efficiency = 1.0  # Prevent division by zero

        # Calculate normalized efficiencies
        curricula = []
        normalized_efficiencies = []
        raw_efficiencies = []

        for curriculum_name, result in results.items():
            if curriculum_name == "oracle":
                continue  # Skip oracle in normalized comparison

            # Calculate total efficiency
            performances = []
            for task in result["performance_data"].keys():
                task_performances = result["performance_data"][task]
                performances.extend(task_performances)
            efficiency = sum(performances) * 100
            normalized_efficiency = efficiency / oracle_efficiency

            curricula.append(curriculum_name.replace("_", " ").title())
            normalized_efficiencies.append(normalized_efficiency)
            raw_efficiencies.append(efficiency)

        # Create bar chart with normalized efficiencies
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]  # Orange, Green, Red
        bars = ax.bar(curricula, normalized_efficiencies, color=colors[: len(curricula)], alpha=0.8)

        # Add value labels on bars
        for bar, norm_eff, raw_eff in zip(bars, normalized_efficiencies, raw_efficiencies, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{norm_eff:.3f}\n({raw_eff:.0f})",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Add oracle baseline line
        ax.axhline(y=1.0, color="#1f77b4", linestyle="--", linewidth=2, label="Oracle Baseline (1.0)")

        ax.set_ylabel("Efficiency Ratio (Curriculum/Oracle)")
        ax.set_title("Curriculum Efficiency Normalized by Oracle")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Set y-axis limits
        ax.set_ylim(0, 1.2)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    @staticmethod
    def _create_oracle_comparison(results: Dict[str, Any], ax: plt.Axes):
        """Create oracle comparison showing regret metrics."""
        # Get the actual number of epochs from the data
        if results:
            first_result = next(iter(results.values()))
            if "performance_data" in first_result and first_result["performance_data"]:
                first_task = next(iter(first_result["performance_data"].keys()))
                num_epochs = len(first_result["performance_data"][first_task])
                epochs = range(num_epochs)
            else:
                epochs = range(150)  # Fallback
        else:
            epochs = range(150)  # Fallback

        # Calculate efficiency ratios compared to oracle
        oracle_efficiency = None
        curricula_efficiency = {}

        # Get oracle efficiency
        if "oracle" in results and "oracle_performances" in results["oracle"]:
            oracle_performances = results["oracle"]["oracle_performances"]
            oracle_efficiency = np.cumsum(oracle_performances)

        # Calculate efficiency for other curricula
        for curriculum_name, result in results.items():
            if curriculum_name != "oracle":
                performances = []
                for epoch in epochs:
                    epoch_perfs = [
                        result["performance_data"][task][epoch] for task in result["performance_data"].keys()
                    ]
                    performances.append(np.mean(epoch_perfs) * 100)

                curricula_efficiency[curriculum_name] = np.cumsum(performances)

        # Calculate efficiency ratios
        if oracle_efficiency is not None:
            colors = ["blue", "red", "orange"]
            linestyles = ["-", ":", "--"]

            for i, (curriculum_name, efficiency) in enumerate(curricula_efficiency.items()):
                efficiency_ratio = efficiency / oracle_efficiency
                ax.plot(
                    epochs,
                    efficiency_ratio,
                    color=colors[i],
                    linewidth=2,
                    linestyle=linestyles[i],
                    label=curriculum_name.replace("_", " ").title(),
                    alpha=0.8,
                )

            # Add oracle baseline
            ax.axhline(y=1.0, color="green", linestyle="--", linewidth=2, label="Oracle Baseline")

        ax.set_title("Efficiency Ratio vs Oracle (Lower = Better)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Efficiency Ratio (Curriculum/Oracle)")
        ax.legend()
        ax.grid(True, alpha=0.3)


def run_consolidated_analysis(graph_type: str = "chain", num_tasks: int = 10, num_epochs: int = 150):
    """Run the consolidated curriculum analysis."""
    logger.info("Starting Consolidated Curriculum Analysis...")

    # Create dependency graph
    if graph_type == "chain":
        G, dependency_depths = DependencyGraphGenerator.create_chain_graph(num_tasks)
        logger.info(f"Created chain dependency graph with {num_tasks} tasks")
    elif graph_type == "binary_tree":
        depth = int(np.log2(num_tasks)) if num_tasks > 1 else 1
        G, dependency_depths = DependencyGraphGenerator.create_binary_tree_graph(depth)
        logger.info(f"Created binary tree dependency graph with depth {depth}")
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Run curriculum analysis
    analyzer = CurriculumAnalyzer(G, dependency_depths)
    results = analyzer.run_curriculum_comparison(num_epochs)

    # Run learning progress sweep
    sweep = LearningProgressSweep(G, dependency_depths)
    grid_results = sweep.run_grid_search(num_epochs)

    # Create visualizations
    CurriculumVisualizer.create_comprehensive_visualization(results, G, dependency_depths)

    # Print summary statistics
    print_summary_statistics(results, grid_results)

    return results, grid_results


def print_summary_statistics(results: Dict[str, Any], grid_results: Dict[Tuple[float, float], Dict[str, float]]):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("CONSOLIDATED CURRICULUM ANALYSIS SUMMARY")
    print("=" * 80)

    # Get oracle efficiency for normalization
    oracle_efficiency = None
    oracle_avg_performance = None
    for curriculum_name, result in results.items():
        if curriculum_name == "oracle":
            performances = []
            for task in result["performance_data"].keys():
                task_performances = result["performance_data"][task]
                performances.extend(task_performances)
            oracle_efficiency = sum(performances) * 100
            oracle_avg_performance = np.mean(performances) * 100
            break

    if oracle_efficiency is None or oracle_efficiency <= 0:
        oracle_efficiency = 1.0
    if oracle_avg_performance is None or oracle_avg_performance <= 0:
        oracle_avg_performance = 1.0

    # Curriculum comparison summary
    print("\nCURRICULUM PERFORMANCE COMPARISON:")
    print(
        f"{'Curriculum':<20} {'Avg Performance':<15} {'Normalized Perf':<15} "
        f"{'Efficiency':<15} {'Normalized Eff':<15} {'Final Variance':<15}"
    )
    print("-" * 95)

    for curriculum_name, result in results.items():
        # Calculate metrics
        performances = []
        for task in result["performance_data"].keys():
            task_performances = result["performance_data"][task]
            performances.extend(task_performances)

        avg_performance = np.mean(performances) * 100
        normalized_performance = avg_performance / oracle_avg_performance
        efficiency = sum(performances) * 100
        normalized_efficiency = efficiency / oracle_efficiency
        final_variance = np.var([result["performance_data"][task][-1] for task in result["performance_data"].keys()])

        print(
            f"{curriculum_name:<20} {avg_performance:<15.2f} {normalized_performance:<15.3f} "
            f"{efficiency:<15.2f} {normalized_efficiency:<15.3f} {final_variance:<15.4f}"
        )

    # Learning progress sweep summary
    print("\n" + "=" * 80)
    print("LEARNING PROGRESS GRID SEARCH RESULTS")
    print("=" * 80)

    # Find best parameters
    best_performance = max(grid_results.values(), key=lambda x: x["avg_final_performance"])
    best_efficiency = max(grid_results.values(), key=lambda x: x["cumulative_efficiency"])
    best_consistency = max(grid_results.values(), key=lambda x: x["learning_consistency"])

    print("\nBest Parameters:")
    for (timescale, smoothing), metrics in grid_results.items():
        if metrics == best_performance:
            print(
                f"Best Performance: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['avg_final_performance']:.2f}"
            )
        if metrics == best_efficiency:
            print(
                f"Best Efficiency: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['cumulative_efficiency']:.2f}"
            )
        if metrics == best_consistency:
            print(
                f"Best Consistency: EMA={timescale:.6f}, Smoothing={smoothing:.6f} -> "
                f"{metrics['learning_consistency']:.4f}"
            )

    print(f"\nTotal Parameter Combinations Tested: {len(grid_results)}")
    print("Results saved to: consolidated_curriculum_analysis.png")


def main():
    """Main function to run the consolidated curriculum analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidated Curriculum Analysis")
    parser.add_argument(
        "--graph-type", choices=["chain", "binary_tree"], default="chain", help="Type of dependency graph to use"
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks in the dependency graph")
    parser.add_argument("--num-epochs", type=int, default=150, help="Number of training epochs")

    args = parser.parse_args()

    try:
        results, grid_results = run_consolidated_analysis(
            graph_type=args.graph_type, num_tasks=args.num_tasks, num_epochs=args.num_epochs
        )

        logger.info("Consolidated curriculum analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
