"""
Enhanced Topological Oracle for Curriculum Analysis

This module implements a realistic oracle curriculum that uses:
1. Actual task dependency graphs
2. Realistic learning curves based on task difficulty
3. Topological sorting with optimization within dependency levels
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearningCurve:
    """Represents a realistic learning curve for a task."""

    task_id: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    max_performance: float  # Maximum achievable performance
    learning_rate: float  # How quickly performance improves
    plateau_threshold: float  # Performance level where learning plateaus
    noise_scale: float  # Amount of performance variance

    def predict_performance(self, training_time: int, base_performance: float = 0.0) -> float:
        """
        Predict performance at a given training time.

        Args:
            training_time: Number of training epochs
            base_performance: Starting performance level

        Returns:
            Predicted performance (0.0 to 1.0)
        """
        # Ensure minimum baseline performance
        min_performance = max(0.1, base_performance)

        # Sigmoid learning curve with plateau effect - much slower learning
        learning_progress = 1 / (1 + np.exp(-(training_time - self.learning_rate * 30) / (self.learning_rate * 2)))

        # Plateau effect - performance growth slows down much later
        plateau_effect = 1 / (1 + np.exp(-(training_time - self.plateau_threshold * 40) / 10))

        # Calculate final performance
        performance = min_performance + (self.max_performance - min_performance) * learning_progress * plateau_effect

        # Add realistic noise
        noise = np.random.normal(0, self.noise_scale)
        performance += noise

        return np.clip(performance, 0.1, 1.0)  # Ensure minimum 0.1 performance

    def calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency (performance gain per time unit)."""
        # Estimate efficiency as peak performance divided by time to reach 80% of peak
        time_to_80_percent = self.learning_rate * 15  # Approximate
        return self.max_performance / max(time_to_80_percent, 1.0)


class EnhancedTopologicalOracle:
    """
    Enhanced oracle using topological sort with learning curve optimization.

    This oracle:
    1. Respects task dependencies (topological ordering)
    2. Optimizes within each dependency level using learning curves
    3. Provides realistic performance predictions
    """

    def __init__(self, tasks: Dict[str, List[str]], learning_curves: Dict[str, LearningCurve]):
        """
        Initialize the enhanced oracle.

        Args:
            tasks: Dictionary mapping task_id to list of dependencies
            learning_curves: Dictionary mapping task_id to LearningCurve
        """
        self.tasks = tasks
        self.learning_curves = learning_curves
        self.dependency_graph = self._build_dependency_graph()
        self.optimal_order = self._compute_optimal_order()
        self.performance_cache = {}

        logger.info(f"Enhanced Topological Oracle initialized with {len(tasks)} tasks")

    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build the task dependency graph."""
        G = nx.DiGraph()

        # Add nodes
        for task_id in self.tasks.keys():
            G.add_node(task_id)

        # Add edges (dependencies)
        for task_id, dependencies in self.tasks.items():
            for dep in dependencies:
                G.add_edge(dep, task_id)

        # Check for cycles
        try:
            nx.find_cycle(G)
            raise ValueError("Task dependency graph contains cycles!")
        except nx.NetworkXNoCycle:
            pass

        return G

    def _get_dependency_levels(self) -> Dict[int, List[str]]:
        """Get tasks grouped by dependency level."""
        levels = {}

        for task in nx.topological_sort(self.dependency_graph):
            level = len(list(nx.ancestors(self.dependency_graph, task)))
            if level not in levels:
                levels[level] = []
            levels[level].append(task)

        return levels

    def _calculate_learning_efficiency(self, task_id: str) -> float:
        """Calculate learning efficiency for a task."""
        if task_id not in self.learning_curves:
            # Default efficiency for tasks without learning curves
            return 1.0

        return self.learning_curves[task_id].calculate_learning_efficiency()

    def _compute_optimal_order(self) -> List[str]:
        """Compute optimal task ordering using topological sort + optimization."""
        levels = self._get_dependency_levels()

        optimal_order = []
        for level in sorted(levels.keys()):
            level_tasks = levels[level]

            # Sort by learning efficiency (higher efficiency first)
            level_tasks.sort(key=lambda t: self._calculate_learning_efficiency(t), reverse=True)
            optimal_order.extend(level_tasks)

        logger.info(f"Optimal task order: {optimal_order}")
        return optimal_order

    def predict_performance(self, epoch: int, completed_tasks: set) -> float:
        """
        Predict oracle performance at a given epoch.

        Args:
            epoch: Current training epoch
            completed_tasks: Set of completed task IDs

        Returns:
            Predicted oracle performance (0.0 to 1.0)
        """
        if not completed_tasks:
            return 0.1  # Small baseline performance instead of 0

        # Calculate performance for each completed task
        total_performance = 0.0
        valid_tasks = 0

        for task_id in completed_tasks:
            if task_id in self.learning_curves:
                # Calculate how long this task has been trained
                task_position = self.optimal_order.index(task_id)
                training_time = max(0, epoch - task_position * 15)  # Assume 15 epochs per task

                # Get base performance from dependencies
                dependencies = self.tasks.get(task_id, [])
                base_performance = 0.0
                if dependencies:
                    # Base performance is average of dependency performances
                    dep_performances = []
                    for dep in dependencies:
                        if dep in completed_tasks and dep in self.learning_curves:
                            dep_training_time = max(0, epoch - self.optimal_order.index(dep) * 15)
                            dep_perf = self.learning_curves[dep].predict_performance(dep_training_time)
                            dep_performances.append(dep_perf)

                    if dep_performances:
                        base_performance = np.mean(dep_performances)

                # Predict task performance
                task_performance = self.learning_curves[task_id].predict_performance(training_time, base_performance)
                total_performance += task_performance
                valid_tasks += 1

        if valid_tasks == 0:
            return 0.1  # Small baseline performance instead of 0

        return total_performance / valid_tasks

    def get_optimal_curriculum_performance(self, epochs: int) -> List[float]:
        """
        Get optimal curriculum performance over all epochs.

        Args:
            epochs: Number of training epochs

        Returns:
            List of performance values for each epoch (scaled to 0-100)
        """
        performances = []

        for epoch in range(epochs):
            # Determine which tasks are completed by this epoch
            completed_tasks = set()
            for i, task_id in enumerate(self.optimal_order):
                # Assume tasks are completed in order with more separation
                completion_epoch = i * 15  # 15 epochs per task (3x slower)
                if epoch >= completion_epoch:
                    completed_tasks.add(task_id)

            performance = self.predict_performance(epoch, completed_tasks)
            # Scale to 0-100 range for consistency with other curricula
            performances.append(performance * 100)

        return performances

    def get_task_completion_schedule(self, epochs: int) -> Dict[str, List[int]]:
        """
        Get the schedule of when each task is completed.

        Args:
            epochs: Number of training epochs

        Returns:
            Dictionary mapping task_id to list of completion epochs
        """
        schedule = {}

        for i, task_id in enumerate(self.optimal_order):
            # Calculate when this task is completed
            completion_epoch = i * 15  # 15 epochs per task (3x slower)
            if completion_epoch < epochs:
                schedule[task_id] = [completion_epoch]

        return schedule


def create_realistic_learning_curves(tasks: Dict[str, List[str]]) -> Dict[str, LearningCurve]:
    """
    Create realistic learning curves for tasks based on their dependencies.

    Args:
        tasks: Dictionary mapping task_id to list of dependencies

    Returns:
        Dictionary mapping task_id to LearningCurve
    """
    learning_curves = {}

    for task_id, dependencies in tasks.items():
        # Calculate difficulty based on actual dependency depth (not just count)
        # For sequential chain: A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9
        if task_id in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            # Use position in alphabet for sequential chain
            dependency_depth = ord(task_id) - ord("A")
        else:
            # Fallback to dependency count for other task structures
            dependency_depth = len(dependencies)

        difficulty = min(0.9, 0.1 + dependency_depth * 0.1)  # 0.1 to 1.0

        # Adjust parameters based on difficulty - make learning much slower
        max_performance = 0.95 - difficulty * 0.2  # 0.95 to 0.75
        learning_rate = 0.3 - difficulty * 0.15  # 0.3 to 0.15 (much slower)
        plateau_threshold = 0.4 + difficulty * 0.4  # 0.4 to 0.8 (later plateau)
        noise_scale = 0.02 + difficulty * 0.03  # 0.02 to 0.05

        learning_curves[task_id] = LearningCurve(
            task_id=task_id,
            difficulty=difficulty,
            max_performance=max_performance,
            learning_rate=learning_rate,
            plateau_threshold=plateau_threshold,
            noise_scale=noise_scale,
        )

    return learning_curves


def create_enhanced_oracle_from_demo_tasks() -> EnhancedTopologicalOracle:
    """
    Create an enhanced oracle using the demo task structure (A->B->C->...->J).

    Returns:
        EnhancedTopologicalOracle instance
    """
    # Define the sequential chain of tasks
    tasks = {
        "A": [],  # Root task
        "B": ["A"],  # Depends on A
        "C": ["B"],  # Depends on B
        "D": ["C"],  # Depends on C
        "E": ["D"],  # Depends on D
        "F": ["E"],  # Depends on E
        "G": ["F"],  # Depends on F
        "H": ["G"],  # Depends on G
        "I": ["H"],  # Depends on H
        "J": ["I"],  # Depends on I
    }

    # Create realistic learning curves
    learning_curves = create_realistic_learning_curves(tasks)

    return EnhancedTopologicalOracle(tasks, learning_curves)
