"""
Dynamical Curriculum Analysis using Simplified Curriculum Implementations

This module provides simplified curriculum implementations that simulate
the behavior of the real curricula from the main branch, while providing
a simple dynamical environment for testing curriculum performance.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearningDynamicsConfig:
    """Configuration for the learning dynamics system."""

    # Learning dynamics parameters
    gamma: float = 0.3  # Parent task contribution to child learning
    lambda_forget: float = 0.01  # Forgetting rate
    dt: float = 0.1  # Time step for numerical integration

    # Task sampling parameters
    samples_per_epoch: int = 3  # Number of task samples per epoch

    # Performance threshold for success metrics
    success_threshold: float = 0.9

    # Learning progress parameters (for grid search)
    ema_timescale: float = 0.001
    progress_smoothing: float = 0.05

    # Focused learning incentives
    focus_bonus: float = 0.0  # Bonus for repeated sampling of same task
    focus_decay: float = 0.1  # Decay rate for focus bonus
    specialization_threshold: float = 0.7  # Performance threshold for specialization benefits
    specialization_bonus: float = 0.0  # Bonus for high performance on focused tasks
    exploration_penalty: float = 0.0  # Penalty for switching between tasks
    dependency_strength: float = 1.0  # Strength of dependency effects (0 = no dependencies)


@dataclass
class GridSearchConfig:
    """Configuration for grid search over learning progress parameters."""

    # Grid search parameters
    ema_timescales: List[float] = None  # List of ema_timescale values to test
    progress_smoothings: List[float] = None  # List of progress_smoothing values to test
    num_epochs: int = 150  # Number of epochs for each grid search run

    def __post_init__(self):
        if self.ema_timescales is None:
            self.ema_timescales = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        if self.progress_smoothings is None:
            self.progress_smoothings = [0.001, 0.01, 0.05, 0.1, 0.2]


class DynamicalTaskEnvironment:
    """
    A structured task environment with explicit learning dynamics.

    The performance on each task evolves according to differential equations
    that model dependencies, learning progress, and forgetting.
    """

    def __init__(self, dependency_graph: nx.DiGraph, config: LearningDynamicsConfig):
        """
        Initialize the dynamical task environment.

        Args:
            dependency_graph: DAG defining task dependencies
            config: Configuration for learning dynamics
        """
        self.graph = dependency_graph
        self.config = config
        self.tasks = list(dependency_graph.nodes())
        self.num_tasks = len(self.tasks)

        # Initialize performance state
        self.performance = {task: 0.1 for task in self.tasks}  # Start at 10% performance
        self.performance_history = {task: [0.1] for task in self.tasks}

        # Initialize sampling counters
        self.sampling_counts = {task: 0 for task in self.tasks}

        # Initialize focus tracking
        self.focus_levels = {task: 0.0 for task in self.tasks}  # Focus level for each task
        self.last_sampled_tasks = []  # Track recent sampling history

        # Validate graph is a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Dependency graph must be a DAG")

        logger.info(f"Initialized dynamical environment with {self.num_tasks} tasks")

    def get_parents(self, task: str) -> List[str]:
        """Get parent tasks (prerequisites) for a given task."""
        return list(self.graph.predecessors(task))

    def get_children(self, task: str) -> List[str]:
        """Get child tasks (dependents) for a given task."""
        return list(self.graph.successors(task))

    def sample_tasks(self, curriculum, num_samples: int) -> List[str]:
        """
        Sample tasks using the provided curriculum.

        Args:
            curriculum: Curriculum to use for sampling
            num_samples: Number of tasks to sample

        Returns:
            List of sampled task names
        """
        sampled_tasks = []

        for _ in range(num_samples):
            # Get task from curriculum
            task_id = curriculum.get_task()
            sampled_tasks.append(task_id)

        return sampled_tasks

    def update_sampling_counts(self, sampled_tasks: List[str]):
        """Update the sampling counts for the given tasks."""
        for task in sampled_tasks:
            if task in self.sampling_counts:
                self.sampling_counts[task] += 1

        # Update sampling history for exploration penalty
        self.last_sampled_tasks.extend(sampled_tasks)
        # Keep only last 10 samples to avoid memory growth
        if len(self.last_sampled_tasks) > 10:
            self.last_sampled_tasks = self.last_sampled_tasks[-10:]

    def step_dynamics(self):
        """
        Step the learning dynamics forward using numerical integration.

        Enhanced differential equation that rewards focused learning:
        dP_i/dt = (S_i + γ * Σ_{j∈parents(i)} S_j) * (Π_{c∈children(i)} P_c) * (1 - P_i) *
        (1 + focus_bonus_i) * (1 + specialization_bonus_i) - λ * P_i - exploration_penalty_i
        """
        # Calculate the derivative for each task
        derivatives = {}

        for task in self.tasks:
            # Get sampling count for this task
            S_i = self.sampling_counts[task]

            # Calculate parent contribution (scaled by dependency strength)
            parent_contribution = sum(self.sampling_counts[parent] for parent in self.get_parents(task))
            parent_contribution *= self.config.dependency_strength

            # Calculate children gating term (scaled by dependency strength)
            children = self.get_children(task)
            if children and self.config.dependency_strength > 0:
                children_gating = np.prod([self.performance[child] for child in children])
                children_gating = 1.0 + self.config.dependency_strength * (children_gating - 1.0)
            else:
                children_gating = 1.0  # No children means no gating

            # Calculate saturation term
            saturation = 1.0 - self.performance[task]

            # Calculate focus bonus
            focus_bonus = self.config.focus_bonus * self.focus_levels[task]

            # Calculate specialization bonus (bonus for high performance on focused tasks)
            if self.performance[task] >= self.config.specialization_threshold and self.focus_levels[task] > 0.5:
                specialization_bonus = self.config.specialization_bonus * self.focus_levels[task]
            else:
                specialization_bonus = 0.0

            # Calculate exploration penalty (penalty for switching between tasks)
            exploration_penalty = 0.0
            if len(self.last_sampled_tasks) >= 2:
                unique_tasks = len(set(self.last_sampled_tasks[-2:]))
                if unique_tasks > 1:  # If we switched tasks
                    exploration_penalty = self.config.exploration_penalty * S_i

            # Calculate forgetting term
            forgetting = self.config.lambda_forget * self.performance[task]

            # Compute enhanced derivative
            base_learning = (S_i + self.config.gamma * parent_contribution) * children_gating * saturation
            focus_multiplier = (1.0 + focus_bonus) * (1.0 + specialization_bonus)
            derivative = base_learning * focus_multiplier - forgetting - exploration_penalty
            derivatives[task] = derivative

        # Update performance using Euler integration
        for task in self.tasks:
            new_performance = self.performance[task] + self.config.dt * derivatives[task]
            new_performance = np.clip(new_performance, 0.0, 1.0)  # Clamp to [0, 1]

            self.performance[task] = new_performance
            self.performance_history[task].append(new_performance)

        # Update focus levels
        self._update_focus_levels()

        # Reset sampling counts for next epoch
        self.sampling_counts = {task: 0 for task in self.tasks}

    def _update_focus_levels(self):
        """Update focus levels based on recent sampling patterns."""
        # Decay all focus levels
        for task in self.tasks:
            self.focus_levels[task] *= 1.0 - self.config.focus_decay

        # Increase focus for recently sampled tasks
        for task in self.sampling_counts:
            if self.sampling_counts[task] > 0:
                self.focus_levels[task] += self.sampling_counts[task] * 0.1
                self.focus_levels[task] = min(1.0, self.focus_levels[task])  # Cap at 1.0

    def complete_task(self, task_id: str, score: float):
        """Complete a task and update curriculum."""
        # Update performance directly based on score
        if task_id in self.performance:
            self.performance[task_id] = max(self.performance[task_id], score)

    def get_average_performance(self) -> float:
        """Get the average performance across all tasks."""
        return np.mean(list(self.performance.values()))

    def get_performance_vector(self) -> np.ndarray:
        """Get the current performance as a vector."""
        return np.array([self.performance[task] for task in self.tasks])

    def all_tasks_above_threshold(self, threshold: float = None) -> bool:
        """Check if all tasks are above the given threshold."""
        if threshold is None:
            threshold = self.config.success_threshold
        return all(self.performance[task] >= threshold for task in self.tasks)

    def get_time_to_threshold(self, threshold: float = None) -> Optional[int]:
        """Get the epoch when all tasks first exceeded the threshold."""
        if threshold is None:
            threshold = self.config.success_threshold

        for epoch in range(len(self.performance_history[self.tasks[0]])):
            all_above = True
            for task in self.tasks:
                if self.performance_history[task][epoch] < threshold:
                    all_above = False
                    break
            if all_above:
                return epoch
        return None

    def get_learning_efficiency(self) -> float:
        """Calculate learning efficiency as area under the performance curve."""
        # Use the average performance over time
        avg_performance = []
        for epoch in range(len(self.performance_history[self.tasks[0]])):
            epoch_avg = np.mean([self.performance_history[task][epoch] for task in self.tasks])
            avg_performance.append(epoch_avg)

        # Calculate area under curve using trapezoidal rule
        return np.trapz(avg_performance)

    def reset(self):
        """Reset the environment to initial state."""
        self.performance = {task: 0.1 for task in self.tasks}
        self.performance_history = {task: [0.1] for task in self.tasks}
        self.sampling_counts = {task: 0 for task in self.tasks}
        self.focus_levels = {task: 0.0 for task in self.tasks}
        self.last_sampled_tasks = []


class MockCurriculum:
    """Base class for mock curriculum implementations."""

    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.num_tasks = len(task_names)

    def get_task(self) -> str:
        """Get a task from the curriculum."""
        raise NotImplementedError

    def complete_task(self, task_id: str, score: float):
        """Complete a task and update curriculum state."""
        pass


class MockRandomCurriculum(MockCurriculum):
    """Mock random curriculum with uniform sampling."""

    def get_task(self) -> str:
        """Uniform random sampling."""
        return random.choice(self.task_names)


class MockLearningProgressCurriculum(MockCurriculum):
    """Mock learning progress curriculum using EMA difference."""

    def __init__(self, task_names: List[str], ema_timescale: float = 0.001, progress_smoothing: float = 0.05):
        super().__init__(task_names)
        self.ema_timescale = ema_timescale
        self.progress_smoothing = progress_smoothing

        # Initialize EMA tracking
        self.ema_fast = {task: 0.1 for task in task_names}
        self.ema_slow = {task: 0.1 for task in task_names}
        self.performance_history = {task: [0.1] for task in task_names}

    def get_task(self) -> str:
        """Sample based on learning progress."""
        # Calculate learning progress for each task
        progress = {}
        for task in self.task_names:
            if len(self.performance_history[task]) >= 2:
                # Simple learning progress: difference between fast and slow EMA
                progress[task] = abs(self.ema_fast[task] - self.ema_slow[task])
            else:
                progress[task] = 0.0

        # Apply softmax to get probabilities
        progress_values = list(progress.values())
        if sum(progress_values) == 0:
            # If no progress, use uniform distribution
            return random.choice(self.task_names)

        # Softmax with temperature
        exp_progress = [np.exp(p / self.progress_smoothing) for p in progress_values]
        total_exp = sum(exp_progress)
        probs = [exp_p / total_exp for exp_p in exp_progress]

        # Sample based on probabilities
        return np.random.choice(self.task_names, p=probs)

    def complete_task(self, task_id: str, score: float):
        """Update EMAs when task is completed."""
        if task_id in self.task_names:
            # Update performance history
            self.performance_history[task_id].append(score)

            # Update EMAs
            self.ema_fast[task_id] = (1 - self.ema_timescale) * self.ema_fast[task_id] + self.ema_timescale * score
            self.ema_slow[task_id] = (1 - self.ema_timescale * 0.1) * self.ema_slow[
                task_id
            ] + self.ema_timescale * 0.1 * score


class MockPrioritizeRegressedCurriculum(MockCurriculum):
    """Mock prioritize regressed curriculum."""

    def __init__(self, task_names: List[str], moving_avg_decay_rate: float = 0.01):
        super().__init__(task_names)
        self.moving_avg_decay_rate = moving_avg_decay_rate

        # Initialize tracking
        self.reward_averages = {task: 0.1 for task in task_names}
        self.reward_maxes = {task: 0.1 for task in task_names}
        self.task_weights = {task: 1.0 for task in task_names}

    def get_task(self) -> str:
        """Sample based on regression priority."""
        # Normalize weights
        total_weight = sum(self.task_weights.values())
        if total_weight > 0:
            probs = [self.task_weights[task] / total_weight for task in self.task_names]
        else:
            probs = [1.0 / self.num_tasks] * self.num_tasks

        return np.random.choice(self.task_names, p=probs)

    def complete_task(self, task_id: str, score: float):
        """Update regression tracking when task is completed."""
        if task_id in self.task_names:
            # Update moving average
            self.reward_averages[task_id] = (1 - self.moving_avg_decay_rate) * self.reward_averages[
                task_id
            ] + self.moving_avg_decay_rate * score

            # Update maximum
            self.reward_maxes[task_id] = max(self.reward_maxes[task_id], score)

            # Update weights based on regression
            self.task_weights[task_id] = 1e-6 + self.reward_maxes[task_id] / (self.reward_averages[task_id] + 1e-6)


class MockOracleCurriculum(MockCurriculum):
    """Mock oracle curriculum that intelligently selects tasks based on performance and dependencies."""

    def __init__(self, task_names: List[str], dependency_graph: nx.DiGraph):
        super().__init__(task_names)
        self.dependency_graph = dependency_graph
        # Compute optimal ordering using topological sort
        self.optimal_order = list(nx.topological_sort(dependency_graph))

        # Track current performance for intelligent decision making
        self.current_performance = {task: 0.1 for task in task_names}
        self.task_priorities = {task: 0.0 for task in task_names}

        # Oracle parameters
        self.mastery_threshold = 0.85  # Consider task mastered at 85%
        self.focus_threshold = 0.6  # Focus on tasks above 60% performance
        self.transfer_bonus = 2.0  # Bonus for tasks that help others

    def update_performance(self, performance_dict: Dict[str, float]):
        """Update the oracle's knowledge of current performance."""
        self.current_performance.update(performance_dict)
        self._update_priorities()

    def _update_priorities(self):
        """Update task priorities based on current performance and dependencies."""
        for task in self.task_names:
            priority = 0.0

            # Base priority: inverse of current performance (lower performance = higher priority)
            current_perf = self.current_performance[task]
            if current_perf < self.mastery_threshold:
                priority += (self.mastery_threshold - current_perf) * 10.0

            # Dependency priority: focus on prerequisites first
            parents = list(self.dependency_graph.predecessors(task))
            if parents:
                # Check if all parents are mastered
                parent_performance = [self.current_performance[parent] for parent in parents]
                min_parent_perf = min(parent_performance)
                if min_parent_perf < self.mastery_threshold:
                    # This task should wait for parents
                    priority *= 0.1
                else:
                    # Parents are ready, this task is important
                    priority *= 1.5

            # Transfer learning bonus: tasks that help many children
            children = list(self.dependency_graph.successors(task))
            if children:
                # Check how many children are struggling
                struggling_children = sum(
                    1 for child in children if self.current_performance[child] < self.focus_threshold
                )
                priority += struggling_children * self.transfer_bonus

            # Critical path priority: tasks in the optimal order get bonus
            if task in self.optimal_order:
                optimal_idx = self.optimal_order.index(task)
                priority += (len(self.optimal_order) - optimal_idx) * 0.5

            self.task_priorities[task] = max(0.0, priority)

    def get_task(self) -> str:
        """Intelligently select the best task based on current state."""
        # Update priorities based on current performance
        self._update_priorities()

        # Find tasks that are ready to be trained (all parents mastered)
        ready_tasks = []
        for task in self.task_names:
            parents = list(self.dependency_graph.predecessors(task))
            if not parents or all(self.current_performance[parent] >= self.mastery_threshold for parent in parents):
                ready_tasks.append(task)

        if not ready_tasks:
            # If no tasks are ready, pick the one with highest priority anyway
            ready_tasks = self.task_names

        # Select from ready tasks based on priority
        if ready_tasks:
            # Use softmax to convert priorities to probabilities
            priorities = [self.task_priorities[task] for task in ready_tasks]
            max_priority = max(priorities) if priorities else 1.0

            # Normalize and apply softmax
            if max_priority > 0:
                exp_priorities = [np.exp((p - max_priority) / 0.1) for p in priorities]
                total_exp = sum(exp_priorities)
                probs = [exp_p / total_exp for exp_p in exp_priorities]
            else:
                # Uniform distribution if all priorities are 0
                probs = [1.0 / len(ready_tasks)] * len(ready_tasks)

            return np.random.choice(ready_tasks, p=probs)
        else:
            # Fallback to random selection
            return random.choice(self.task_names)

    def complete_task(self, task_id: str, score: float):
        """Update performance knowledge when task is completed."""
        if task_id in self.task_names:
            self.current_performance[task_id] = max(self.current_performance[task_id], score)


class DynamicalCurriculumAnalysis:
    """
    Main analysis class for comparing curricula using the dynamical system.
    """

    def __init__(
        self,
        dependency_graph: nx.DiGraph,
        config: LearningDynamicsConfig,
        lp_ema_timescale: float = 0.001,
        lp_progress_smoothing: float = 0.05,
    ):
        self.dependency_graph = dependency_graph
        self.config = config
        self.environment = DynamicalTaskEnvironment(dependency_graph, config)

        # Initialize curricula using mock implementations
        self.curricula = {
            "random": MockRandomCurriculum(self.environment.tasks),
            "learning_progress": MockLearningProgressCurriculum(
                self.environment.tasks, lp_ema_timescale, lp_progress_smoothing
            ),
            "prioritize_regressed": MockPrioritizeRegressedCurriculum(self.environment.tasks),
            "oracle": MockOracleCurriculum(self.environment.tasks, dependency_graph),
        }

        logger.info(f"Initialized dynamical curriculum analysis with {len(self.curricula)} curricula")

    def run_curriculum_comparison(self, num_epochs: int) -> Dict[str, Dict]:
        """
        Run comparison of all curricula for the specified number of epochs.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Dictionary containing results for each curriculum
        """
        results = {}

        for curriculum_name, curriculum in self.curricula.items():
            logger.info(f"Running {curriculum_name} curriculum...")

            # Reset environment
            self.environment.reset()

            # Run training
            curriculum_results = self._run_curriculum(curriculum, num_epochs)
            results[curriculum_name] = curriculum_results

        return results

    def _run_curriculum(self, curriculum: MockCurriculum, num_epochs: int) -> Dict:
        """Run a single curriculum for the specified number of epochs."""
        performance_history = []
        sampling_history = []

        for _epoch in range(num_epochs):
            # Update oracle with current performance if it's the oracle curriculum
            if isinstance(curriculum, MockOracleCurriculum):
                curriculum.update_performance(self.environment.performance)

            # Sample tasks
            sampled_tasks = self.environment.sample_tasks(curriculum, self.config.samples_per_epoch)
            self.environment.update_sampling_counts(sampled_tasks)

            # Complete tasks with current performance
            for task in sampled_tasks:
                current_performance = self.environment.performance[task]
                curriculum.complete_task(task, current_performance)

            # Step dynamics
            self.environment.step_dynamics()

            # Record performance
            avg_performance = self.environment.get_average_performance()
            performance_history.append(avg_performance)

            # Record sampling probabilities (simplified)
            sampling_probs = {task: 1.0 / len(self.environment.tasks) for task in self.environment.tasks}
            sampling_history.append(sampling_probs)

        # Calculate metrics
        learning_efficiency = self.environment.get_learning_efficiency()
        time_to_threshold = self.environment.get_time_to_threshold()

        return {
            "performance_history": performance_history,
            "sampling_history": sampling_history,
            "learning_efficiency": learning_efficiency,
            "time_to_threshold": time_to_threshold,
            "final_performance": performance_history[-1] if performance_history else 0.0,
            "final_performance_vector": self.environment.get_performance_vector().tolist(),
        }

    def run_grid_search(self, grid_config: GridSearchConfig) -> Dict[Tuple[float, float], Dict[str, float]]:
        """
        Run grid search over learning progress parameters.

        Args:
            grid_config: Configuration for grid search

        Returns:
            Dictionary mapping (ema_timescale, progress_smoothing) to metrics
        """
        logger.info("Starting Learning Progress Grid Search...")

        grid_results = {}
        total_combinations = len(grid_config.ema_timescales) * len(grid_config.progress_smoothings)
        current_combination = 0

        for ema_timescale in grid_config.ema_timescales:
            for progress_smoothing in grid_config.progress_smoothings:
                current_combination += 1
                logger.info(
                    f"Testing EMA={ema_timescale:.6f}, Smoothing={progress_smoothing:.6f} "
                    f"({current_combination}/{total_combinations})"
                )

                # Create curriculum with these parameters
                curriculum = MockLearningProgressCurriculum(
                    self.environment.tasks, ema_timescale=ema_timescale, progress_smoothing=progress_smoothing
                )

                # Reset environment
                self.environment.reset()

                # Run training
                curriculum_results = self._run_curriculum(curriculum, grid_config.num_epochs)

                # Store results
                grid_results[(ema_timescale, progress_smoothing)] = {
                    "learning_efficiency": curriculum_results["learning_efficiency"],
                    "time_to_threshold": curriculum_results["time_to_threshold"],
                    "final_performance": curriculum_results["final_performance"],
                    "performance_history": curriculum_results["performance_history"],
                }

        return grid_results

    def print_results(self, results: Dict[str, Dict]):
        """Print a summary of the results."""
        print("\n" + "=" * 80)
        print("DYNAMICAL CURRICULUM ANALYSIS RESULTS")
        print("=" * 80)

        print("\nConfiguration:")
        print(f"  Tasks: {self.environment.num_tasks}")
        print(f"  Epochs: {len(results['random']['performance_history'])}")
        print(f"  Gamma: {self.config.gamma}")
        print(f"  Lambda: {self.config.lambda_forget}")
        print(f"  Success threshold: {self.config.success_threshold}")

        print("\nCurriculum Performance Comparison:")
        print(f"{'Curriculum':<20} {'Efficiency':<12} {'Time to Threshold':<18} {'Final Perf':<12}")
        print("-" * 70)

        for curriculum_name, result in results.items():
            efficiency = result["learning_efficiency"]
            time_to_thresh = result["time_to_threshold"]
            final_perf = result["final_performance"]

            time_str = f"{time_to_thresh}" if time_to_thresh is not None else "∞"
            print(f"{curriculum_name:<20} {efficiency:<12.2f} {time_str:<18} {final_perf:<12.3f}")

        print("\n" + "=" * 80)

    def print_grid_search_results(self, grid_results: Dict[Tuple[float, float], Dict[str, float]]):
        """Print a summary of grid search results."""
        print("\n" + "=" * 80)
        print("LEARNING PROGRESS GRID SEARCH RESULTS")
        print("=" * 80)

        # Find best parameters for each metric
        best_efficiency = max(grid_results.items(), key=lambda x: x[1]["learning_efficiency"])
        best_time = min(grid_results.items(), key=lambda x: x[1]["time_to_threshold"] or float("inf"))
        best_performance = max(grid_results.items(), key=lambda x: x[1]["final_performance"])

        print("\nBEST PARAMETER COMBINATIONS:")
        print(
            f"Best Efficiency: EMA={best_efficiency[0][0]:.6f}, "
            f"Smoothing={best_efficiency[0][1]:.6f} -> {best_efficiency[1]['learning_efficiency']:.2f}"
        )
        print(
            f"Best Time to Threshold: EMA={best_time[0][0]:.6f}, "
            f"Smoothing={best_time[0][1]:.6f} -> {best_time[1]['time_to_threshold']}"
        )
        print(
            f"Best Final Performance: EMA={best_performance[0][0]:.6f}, "
            f"Smoothing={best_performance[0][1]:.6f} -> {best_performance[1]['final_performance']:.3f}"
        )

        print("\nPARAMETER SENSITIVITY:")
        print(f"Total combinations tested: {len(grid_results)}")
        min_eff = min(r["learning_efficiency"] for r in grid_results.values())
        max_eff = max(r["learning_efficiency"] for r in grid_results.values())
        print(f"Efficiency range: {min_eff:.2f} - {max_eff:.2f}")
        min_perf = min(r["final_performance"] for r in grid_results.values())
        max_perf = max(r["final_performance"] for r in grid_results.values())
        print(f"Final performance range: {min_perf:.3f} - {max_perf:.3f}")

        print("\n" + "=" * 80)


def create_chain_dependency_graph(num_tasks: int) -> nx.DiGraph:
    """Create a sequential chain dependency graph."""
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_tasks):
        task_id = f"task_{i}"
        G.add_node(task_id)

        # Add edge from previous task (except for first task)
        if i > 0:
            prev_task = f"task_{i - 1}"
            G.add_edge(prev_task, task_id)

    return G


def create_binary_tree_dependency_graph(num_tasks: int) -> nx.DiGraph:
    """Create a binary tree dependency graph."""
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_tasks):
        task_id = f"task_{i}"
        G.add_node(task_id)

        # Add edges to children (binary tree structure)
        left_child = 2 * i + 1
        right_child = 2 * i + 2

        if left_child < num_tasks:
            G.add_edge(task_id, f"task_{left_child}")
        if right_child < num_tasks:
            G.add_edge(task_id, f"task_{right_child}")

    return G


def run_dynamical_analysis(
    num_tasks: int = 10,
    num_epochs: int = 200,
    config: Optional[LearningDynamicsConfig] = None,
    run_grid_search: bool = False,
    grid_config: Optional[GridSearchConfig] = None,
    graph_type: str = "chain",
    lp_ema_timescale: float = 0.001,
    lp_progress_smoothing: float = 0.05,
) -> Dict[str, Dict]:
    """
    Run the complete dynamical curriculum analysis using mock curriculum implementations.

    Args:
        num_tasks: Number of tasks in the dependency graph
        num_epochs: Number of training epochs
        config: Configuration for learning dynamics
        run_grid_search: Whether to run grid search over LP parameters
        grid_config: Configuration for grid search
        graph_type: Type of dependency graph ("chain" or "binary_tree")
        lp_ema_timescale: EMA timescale for learning progress curriculum
        lp_progress_smoothing: Progress smoothing for learning progress curriculum

    Returns:
        Results dictionary for all curricula and grid search (if enabled)
    """
    if config is None:
        config = LearningDynamicsConfig()

    # Create dependency graph
    if graph_type == "chain":
        dependency_graph = create_chain_dependency_graph(num_tasks)
    elif graph_type == "binary_tree":
        dependency_graph = create_binary_tree_dependency_graph(num_tasks)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Create analysis with custom LP parameters
    analysis = DynamicalCurriculumAnalysis(dependency_graph, config, lp_ema_timescale, lp_progress_smoothing)

    # Run comparison
    results = analysis.run_curriculum_comparison(num_epochs)

    # Run grid search if requested
    if run_grid_search:
        if grid_config is None:
            grid_config = GridSearchConfig()

        logger.info("Running learning progress grid search...")
        grid_results = analysis.run_grid_search(grid_config)
        results["grid_search"] = grid_results

        # Print grid search results
        analysis.print_grid_search_results(grid_results)

    # Print results
    analysis.print_results(results)

    return results


if __name__ == "__main__":
    # Run the analysis with grid search
    results = run_dynamical_analysis(
        num_tasks=10,
        num_epochs=150,
        run_grid_search=True,
        grid_config=GridSearchConfig(
            ema_timescales=[0.00001, 0.0001, 0.001, 0.01, 0.1],
            progress_smoothings=[0.001, 0.01, 0.05, 0.1, 0.2],
            num_epochs=100,
        ),
    )
