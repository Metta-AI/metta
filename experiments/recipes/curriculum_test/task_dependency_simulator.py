"""
Task Dependency Learning Simulator.

A clean, consolidated implementation that simulates task dependency learning
dynamics without using mettagrid. Integrates with existing metta infrastructure
for curriculum learning and stats reporting.
"""

import logging
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from metta.cogworks.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.common.tool import Tool
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


class TaskDependencySimulator:
    """
    Simulates task dependency learning dynamics with curriculum-driven task selection.

    This simulator models chains of dependent tasks where parent tasks contribute to
    child task learning, following dynamical system equations for performance updates.
    """

    def __init__(
        self,
        num_tasks: int = 10,
        num_epochs: int = 100,
        samples_per_epoch: int = 50,
        gamma: float = 0.1,  # Parent contribution factor
        lambda_forget: float = 0.1,  # Forgetting rate
        performance_threshold: float = 0.9,
        task_seed: Optional[int] = None,
        dt: float = 0.01,  # Time step scaling for dynamics updates
        task_noise_std: float = 0.1,  # Standard deviation of task-specific noise
    ):
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.gamma = gamma
        self.lambda_forget = lambda_forget
        self.performance_threshold = performance_threshold
        self.task_seed = task_seed or random.randint(0, 2**31 - 1)
        self.dt = dt
        self.task_noise_std = task_noise_std

        # Initialize task dependency chain (0 -> 1 -> 2 -> ...)
        self._build_task_chain()

        # Initialize performance tracking
        self.P = torch.full((num_tasks,), 0.01)  # Current performance
        self.current_epoch = 0
        self.epoch_sample_counts = torch.zeros(num_tasks)
        self.total_sample_counts = torch.zeros(num_tasks)

        # Task-specific noise (generated from seed)
        self._task_noise = self._generate_task_noise()

        # History for analysis
        self.performance_history = [self.P.clone()]
        self.sample_history = []

        # Track individual task rewards for plotting (focus on first task)
        self.task_reward_history = {i: [] for i in range(num_tasks)}
        self.task_sample_numbers = {i: [] for i in range(num_tasks)}

        # Track task 0 sampling and learning progress percentile over time
        self.task_0_cumulative_samples = []
        self.task_0_lp_percentiles = []
        self.task_0_lp_scores = []
        self.epoch_numbers = []

    def _build_task_chain(self) -> None:
        """Build the task dependency chain structure."""
        self.adj = [[] for _ in range(self.num_tasks)]  # children
        self.parents = [[] for _ in range(self.num_tasks)]  # parents

        # Create chain: 0 -> 1 -> 2 -> ... -> (num_tasks-1)
        for i in range(self.num_tasks - 1):
            self.adj[i].append(i + 1)  # i is parent of i+1
            self.parents[i + 1].append(i)  # i+1 has parent i

    def _generate_task_noise(self) -> torch.Tensor:
        """Generate task-specific noise from seed."""
        np.random.seed(self.task_seed)
        task_noise = np.random.normal(0.0, self.task_noise_std, size=self.num_tasks)
        return torch.tensor(task_noise, dtype=torch.float32)

    def reset(self) -> None:
        """Reset simulator state."""
        self.current_epoch = 0
        self.epoch_sample_counts = torch.zeros(self.num_tasks)
        self.total_sample_counts = torch.zeros(self.num_tasks)
        self.P = torch.full((self.num_tasks,), 0.01)
        self.performance_history = [self.P.clone()]
        self.sample_history = []
        self.task_reward_history = {i: [] for i in range(self.num_tasks)}
        self.task_sample_numbers = {i: [] for i in range(self.num_tasks)}
        self.task_0_cumulative_samples = []
        self.task_0_lp_percentiles = []
        self.task_0_lp_scores = []
        self.epoch_numbers = []

    def sample_task(self, task_id: int) -> float:
        """
        Sample a task and return reward based on current performance + noise.

        Args:
            task_id: Task identifier

        Returns:
            Task reward in [0, 1]
        """
        # Update sample count
        self.epoch_sample_counts[task_id] += 1
        self.total_sample_counts[task_id] += 1

        # Calculate reward: base + current performance + task-specific noise
        base_reward = 0.5
        task_reward = (
            base_reward + self.P[task_id].item() + self._task_noise[task_id].item()
        )
        reward = float(np.clip(task_reward, 0.0, 1.0))

        # Track reward history for plotting
        self.task_reward_history[task_id].append(reward)
        self.task_sample_numbers[task_id].append(
            int(self.total_sample_counts[task_id].item())
        )

        return reward

    def complete_epoch(self, curriculum=None) -> Dict[str, Any]:
        """Complete epoch and update task dynamics."""
        # Update task performance based on dynamics
        self._update_task_dynamics()

        # Store history
        self.performance_history.append(self.P.clone())
        self.sample_history.append(self.epoch_sample_counts.clone())

        # Get metrics for logging
        metrics = self._get_epoch_metrics(curriculum)

        # Advance to next epoch and reset sample counts
        self.current_epoch += 1
        self.epoch_sample_counts = torch.zeros(self.num_tasks)

        return metrics

    def _update_task_dynamics(self) -> None:
        """Update task performance based on chain dynamics."""
        current_P = self.P.clone()
        P_dot = torch.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            # Parent contribution
            parent_contribution = sum(
                self.epoch_sample_counts[p] for p in self.parents[i]
            )
            total_stimulus = (
                self.epoch_sample_counts[i] + self.gamma * parent_contribution
            )

            # Children gate: performance gated by children performance
            children_gate = 1.0
            if self.adj[i]:  # If task i has children
                children_gate = torch.prod(
                    torch.tensor([current_P[c] for c in self.adj[i]])
                )

            # Growth and forgetting
            growth = total_stimulus * children_gate * (1 - current_P[i])
            forgetting = self.lambda_forget * current_P[i]
            P_dot[i] = growth - forgetting

        # Update performance (normalized by samples per epoch and scaled by dt)
        new_P = current_P + P_dot * (self.dt / self.samples_per_epoch)
        self.P = torch.clamp(new_P, 0, 1)

    def _get_epoch_metrics(self, curriculum=None) -> Dict[str, Any]:
        """Get current epoch metrics for logging."""
        task_completion_probs = torch.sigmoid((self.P - 0.5) * 4)

        metrics = {
            "task_dependency/epoch": self.current_epoch,
            "task_dependency/mean_performance": self.P.mean().item(),
            "task_dependency/max_performance": self.P.max().item(),
            "task_dependency/min_performance": self.P.min().item(),
            "task_dependency/performance_std": self.P.std().item(),
            "task_dependency/tasks_above_threshold": (
                self.P >= self.performance_threshold
            )
            .sum()
            .item(),
            "task_dependency/total_samples": self.epoch_sample_counts.sum().item(),
        }

        # Add task 0 reward statistics for noise analysis
        if len(self.task_reward_history[0]) > 0:
            task_0_rewards = self.task_reward_history[0]
            metrics.update(
                {
                    "task_0_noise/mean_reward": np.mean(task_0_rewards),
                    "task_0_noise/std_reward": np.std(task_0_rewards),
                    "task_0_noise/min_reward": np.min(task_0_rewards),
                    "task_0_noise/max_reward": np.max(task_0_rewards),
                    "task_0_noise/total_samples": len(task_0_rewards),
                }
            )

        # Add task 0 cumulative sample count (for eviction tracking)
        # This should track the curriculum task that maps to simulator task 0
        if (
            hasattr(self, "_current_task_0_curriculum_id")
            and self._current_task_0_curriculum_id is not None
            and curriculum is not None
            and curriculum._algorithm is not None
        ):
            # Get the actual completion count for the curriculum task 0
            task_stats = curriculum._algorithm.task_tracker.get_task_stats(
                self._current_task_0_curriculum_id
            )
            if task_stats:
                curriculum_samples = task_stats["completion_count"]
                simulator_samples = int(self.total_sample_counts[0].item())
                metrics["task_0_tracking/cumulative_samples"] = curriculum_samples

                # Reduced frequency curriculum vs simulator comparison
                if self.current_epoch % 500 == 0:
                    task_class = self._current_task_0_curriculum_id % self.num_tasks
                    print(
                        f"Epoch {self.current_epoch}: Curriculum task 0 ID {self._current_task_0_curriculum_id} (task class: {task_class}) samples: {curriculum_samples}, Simulator task 0 samples: {simulator_samples}"
                    )
            else:
                metrics["task_0_tracking/cumulative_samples"] = 0
        else:
            # Fallback to simulator task 0 samples if curriculum task not identified yet
            metrics["task_0_tracking/cumulative_samples"] = int(
                self.total_sample_counts[0].item()
            )

        # Add current task 0 LP percentile and score if available
        if len(self.task_0_lp_percentiles) > 0:
            metrics["task_0_tracking/current_lp_percentile"] = (
                self.task_0_lp_percentiles[-1]
            )
        if len(self.task_0_lp_scores) > 0:
            metrics["task_0_tracking/current_lp_score"] = self.task_0_lp_scores[-1]

        # Add individual task metrics for all tasks
        for i in range(self.num_tasks):
            metrics[f"task_dependency/task_{i}_performance"] = self.P[i].item()
            metrics[f"task_dependency/task_{i}_completion_prob"] = (
                task_completion_probs[i].item()
            )
            metrics[f"task_dependency/task_{i}_samples"] = self.epoch_sample_counts[
                i
            ].item()

        # Calculate sampling imbalance metrics
        sampling_imbalance = self._calculate_sampling_imbalance()
        metrics.update(sampling_imbalance)

        return metrics

    def _calculate_sampling_imbalance(self) -> Dict[str, float]:
        """Calculate metrics to quantify how unbalanced sampling is across task classes."""
        import numpy as np

        # Get sample counts for each task class
        total_samples = self.total_sample_counts.numpy().astype(float)

        # Avoid division by zero
        if np.sum(total_samples) == 0:
            return {
                "sampling/coefficient_of_variation": 0.0,
                "sampling/entropy_normalized": 1.0,
                "sampling/max_min_ratio": 1.0,
                "sampling/gini_coefficient": 0.0,
            }

        # Normalize to get proportions
        proportions = total_samples / np.sum(total_samples)
        1.0 / self.num_tasks

        # 1. Coefficient of Variation (CV) - std/mean of sample counts
        # Higher CV = more imbalanced (0 = perfectly balanced)
        mean_samples = np.mean(total_samples)
        std_samples = np.std(total_samples)
        cv = std_samples / mean_samples if mean_samples > 0 else 0.0

        # 2. Normalized Entropy - measures uniformity of distribution
        # Higher entropy = more balanced (1.0 = perfectly uniform, 0 = maximally imbalanced)
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(proportions * np.log(proportions + epsilon))
        max_entropy = np.log(
            self.num_tasks
        )  # Maximum possible entropy (uniform distribution)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 3. Max/Min Ratio - ratio of most sampled to least sampled task
        # Higher ratio = more imbalanced (1.0 = perfectly balanced)
        max_samples = np.max(total_samples)
        min_samples = np.min(total_samples)
        max_min_ratio = max_samples / (min_samples + epsilon)

        # 4. Gini Coefficient - measures inequality
        # Higher Gini = more imbalanced (0 = perfectly balanced, 1 = maximally imbalanced)
        sorted_samples = np.sort(total_samples)
        n = len(sorted_samples)
        if n == 0:
            gini = 0.0
        else:
            cumsum = np.cumsum(sorted_samples)
            gini = (
                (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
            )

        return {
            "sampling/coefficient_of_variation": float(cv),
            "sampling/entropy_normalized": float(normalized_entropy),
            "sampling/max_min_ratio": float(max_min_ratio),
            "sampling/gini_coefficient": float(gini),
        }

    def _get_learning_progress_distributions(self, curriculum) -> Dict[str, Any]:
        """Extract learning progress score distributions from curriculum algorithm."""
        distributions = {}

        try:
            algorithm = curriculum._algorithm
            if algorithm is None:
                return distributions

            # Get all active task IDs
            active_task_ids = list(curriculum._tasks.keys())
            if not active_task_ids:
                return distributions

            # Get learning progress scores for all tasks
            task_scores = algorithm.score_tasks(active_task_ids)

            # Track task 0 percentile in learning progress scores
            self._track_task_0_percentile(task_scores, curriculum)

            if task_scores:
                scores = list(task_scores.values())

                # Overall score distribution statistics
                if scores:
                    distributions.update(
                        {
                            "learning_progress/pool_mean_score": np.mean(scores),
                            "learning_progress/pool_std_score": np.std(scores),
                            "learning_progress/pool_min_score": np.min(scores),
                            "learning_progress/pool_max_score": np.max(scores),
                            "learning_progress/pool_num_tasks": len(scores),
                        }
                    )

                    # Add percentiles
                    percentiles = [25, 50, 75, 90, 95]
                    for p in percentiles:
                        distributions[f"learning_progress/pool_score_p{p}"] = (
                            np.percentile(scores, p)
                        )

                # Group by task dependency position (task position in chain)
                position_scores = {}
                label_scores = {}  # Group by task labels

                for task_id, score in task_scores.items():
                    position = task_id % self.num_tasks  # Position in dependency chain
                    if position not in position_scores:
                        position_scores[position] = []
                    position_scores[position].append(score)

                    # Group by task label (from curriculum task)
                    if task_id in curriculum._tasks:
                        task = curriculum._tasks[task_id]
                        label = getattr(
                            task._env_cfg, "label", f"task_dependency_{task_id}"
                        )
                        # Extract just the position part for consistent labeling
                        if "task_dependency_" in label:
                            label_key = f"task_dep_pos_{position}"
                        else:
                            label_key = label

                        if label_key not in label_scores:
                            label_scores[label_key] = []
                        label_scores[label_key].append(score)

                # Statistics by position in dependency chain
                for position, pos_scores in position_scores.items():
                    if pos_scores:
                        distributions.update(
                            {
                                f"learning_progress/position_{position}_mean_score": np.mean(
                                    pos_scores
                                ),
                                f"learning_progress/position_{position}_std_score": np.std(
                                    pos_scores
                                ),
                                f"learning_progress/position_{position}_count": len(
                                    pos_scores
                                ),
                            }
                        )

                # Statistics by task label
                for label, label_scores_list in label_scores.items():
                    if label_scores_list:
                        distributions.update(
                            {
                                f"learning_progress/label_{label}_mean_score": np.mean(
                                    label_scores_list
                                ),
                                f"learning_progress/label_{label}_std_score": np.std(
                                    label_scores_list
                                ),
                                f"learning_progress/label_{label}_count": len(
                                    label_scores_list
                                ),
                            }
                        )

                # Score distribution bins (histogram-like)
                if scores:
                    score_bins = np.linspace(
                        0, max(scores) if max(scores) > 0 else 1, 10
                    )
                    hist, _ = np.histogram(scores, bins=score_bins)
                    for i, count in enumerate(hist):
                        distributions[f"learning_progress/pool_score_bin_{i}"] = count

                    # Store raw scores for basic statistics (no longer used for histograms)
                    distributions["_learning_progress_scores_raw"] = scores

        except Exception as e:
            logger.warning(f"Failed to extract learning progress distributions: {e}")

        return distributions

    def _track_task_0_percentile(
        self, task_scores: Dict[int, float], curriculum=None
    ) -> None:
        """Track task 0's percentile ranking in learning progress scores."""
        if not task_scores:
            return

        # Find task 0's actual task ID (may be different from 0 due to modulo in simulation)
        task_0_id = None
        for task_id in task_scores.keys():
            if task_id % self.num_tasks == 0:
                task_0_id = task_id
                break

        if task_0_id is None:
            return

        task_0_score = task_scores[task_0_id]
        all_scores = list(task_scores.values())

        # Calculate percentile (what percentage of tasks have lower scores)
        lower_scores = sum(1 for score in all_scores if score < task_0_score)
        percentile = (
            (lower_scores / len(all_scores)) * 100 if len(all_scores) > 0 else 0
        )

        # Get curriculum task 0 sample count
        curriculum_samples = 0
        if curriculum and curriculum._algorithm:
            task_stats = curriculum._algorithm.task_tracker.get_task_stats(task_0_id)
            if task_stats:
                curriculum_samples = task_stats["completion_count"]
                # Periodic task 0 tracking (reduced frequency)
                if self.current_epoch % 500 == 0:  # Log every 500 epochs
                    task_class = task_0_id % self.num_tasks
                    print(
                        f"Epoch {self.current_epoch}: Task 0 curriculum ID {task_0_id} (task class: {task_class}) has {curriculum_samples} cumulative samples"
                    )

        # Track the data
        self.task_0_lp_percentiles.append(percentile)
        self.task_0_lp_scores.append(task_0_score)
        # Store the curriculum task 0 ID for later use
        self._current_task_0_curriculum_id = task_0_id
        # Use curriculum task sample count for plotting
        self.task_0_cumulative_samples.append(curriculum_samples)
        self.epoch_numbers.append(self.current_epoch)

    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.current_epoch >= self.num_epochs

    def get_summary(self) -> Dict[str, Any]:
        """Get final simulation summary."""
        return {
            "final_performances": self.P.tolist(),
            "performance_history": [p.tolist() for p in self.performance_history],
            "sample_history": [s.tolist() for s in self.sample_history],
            "total_samples": self.total_sample_counts.tolist(),
            "num_epochs_completed": self.current_epoch,
            "final_mean_performance": self.P.mean().item(),
            "tasks_above_threshold": (self.P >= self.performance_threshold)
            .sum()
            .item(),
        }


class MockTaskGenerator(TaskGenerator):
    """Task generator that creates minimal configs for task dependency simulation."""

    class Config(TaskGeneratorConfig["MockTaskGenerator"]):
        # No additional config needed - simulator handles all parameters
        pass

    def __init__(self, config: "MockTaskGenerator.Config"):
        super().__init__(config)

    def _generate_task(self, task_id: int, rng) -> MettaGridConfig:
        """Generate a minimal MettaGridConfig for task ID."""
        return MettaGridConfig(label=f"task_dependency_{task_id}")


def create_curriculum(
    num_tasks: int = 10,
    enable_detailed_slice_logging: bool = False,
    ema_timescale: float = 0.1,
    slow_timescale_factor: float = 0.02,
    exploration_bonus: float = 0.1,
    progress_smoothing: float = 0.05,
    use_bidirectional: bool = True,
    max_memory_tasks: int = 100000,
    max_slice_axes: int = 3,
    num_active_tasks: int = 1000,
    rand_task_rate: float = 0.01,
    sampling_temperature: float = 1.0,
    min_presentations_for_eviction: int = 5,
    eviction_threshold_percentile: float = 0.4,
) -> CurriculumConfig:
    """Create curriculum configuration for task dependency simulation."""
    task_gen_config = MockTaskGenerator.Config()

    algorithm_config = LearningProgressConfig(
        use_bidirectional=use_bidirectional,
        ema_timescale=ema_timescale,
        slow_timescale_factor=slow_timescale_factor,
        exploration_bonus=exploration_bonus,
        progress_smoothing=progress_smoothing,
        max_memory_tasks=max_memory_tasks,
        max_slice_axes=max_slice_axes,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        num_active_tasks=num_active_tasks,
        rand_task_rate=rand_task_rate,
        sampling_temperature=sampling_temperature,
        eviction_threshold_percentile=eviction_threshold_percentile,
    )

    return CurriculumConfig(
        task_generator=task_gen_config,
        algorithm_config=algorithm_config,
        num_active_tasks=num_active_tasks,
        min_presentations_for_eviction=min_presentations_for_eviction,
    )


def simulate_task_dependencies(
    num_tasks: int = 10,
    num_epochs: int = 100,
    samples_per_epoch: int = 50,
    gamma: float = 0.1,
    lambda_forget: float = 0.1,
    performance_threshold: float = 0.9,
    task_seed: Optional[int] = None,
    dt: float = 0.1,
    task_noise_std: float = 0.1,
    enable_detailed_slice_logging: bool = False,
    ema_timescale: float = 0.001,
    slow_timescale_factor: float = 0.2,
    exploration_bonus: float = 0.1,
    progress_smoothing: float = 0.05,
    use_bidirectional: bool = True,
    max_memory_tasks: int = 100000,
    max_slice_axes: int = 3,
    num_active_tasks: int = 1000,
    rand_task_rate: float = 0.01,
    sampling_temperature: float = 1.0,
    min_presentations_for_eviction: int = 5,
    eviction_threshold_percentile: float = 0.4,
    wandb_project: str = "metta",
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete task dependency simulation.

    Args:
        num_tasks: Number of tasks in the chain
        num_epochs: Number of training epochs
        samples_per_epoch: Samples per epoch
        gamma: Parent contribution factor
        lambda_forget: Forgetting rate
        performance_threshold: Success threshold
        task_seed: Seed for task-specific noise
        dt: Time step scaling for dynamics updates
        enable_detailed_slice_logging: Enable curriculum slice logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name

    Returns:
        Simulation results dictionary
    """
    # Create simulator
    simulator = TaskDependencySimulator(
        num_tasks=num_tasks,
        num_epochs=num_epochs,
        samples_per_epoch=samples_per_epoch,
        gamma=gamma,
        lambda_forget=lambda_forget,
        performance_threshold=performance_threshold,
        task_seed=task_seed,
        dt=dt,
        task_noise_std=task_noise_std,
    )

    # Create curriculum
    curriculum_config = create_curriculum(
        num_tasks=num_tasks,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        ema_timescale=ema_timescale,
        slow_timescale_factor=slow_timescale_factor,
        exploration_bonus=exploration_bonus,
        progress_smoothing=progress_smoothing,
        use_bidirectional=use_bidirectional,
        max_memory_tasks=max_memory_tasks,
        max_slice_axes=max_slice_axes,
        num_active_tasks=num_active_tasks,
        rand_task_rate=rand_task_rate,
        sampling_temperature=sampling_temperature,
        min_presentations_for_eviction=min_presentations_for_eviction,
        eviction_threshold_percentile=eviction_threshold_percentile,
    )
    curriculum = Curriculum(curriculum_config)

    # Run simulation
    logger.info(f"Starting task dependency simulation for {num_epochs} epochs")
    simulator.reset()
    metrics_history = []

    for epoch in range(num_epochs):
        # Simulate curriculum-driven task sampling
        for _ in range(samples_per_epoch):
            # Get task from curriculum
            task = curriculum.get_task()
            task_id = task._task_id % num_tasks  # Ensure valid task ID

            # Sample the task and get reward
            reward = simulator.sample_task(task_id)

            # Update curriculum with task completion
            task.complete(reward)
            curriculum.update_task_performance(task._task_id, reward)

        # Complete epoch and collect metrics
        epoch_metrics = simulator.complete_epoch(curriculum)
        epoch_metrics.update(curriculum.stats())

        # Process evictions after all samples for the epoch
        evictions_count = curriculum.process_evictions()
        epoch_metrics["curriculum/evictions_this_epoch"] = evictions_count

        # Add learning progress score distributions
        lp_distributions = simulator._get_learning_progress_distributions(curriculum)
        epoch_metrics.update(lp_distributions)

        metrics_history.append(epoch_metrics)

        # Log progress
        if epoch % 10 == 0:
            # Get sampling entropy for logging
            sampling_entropy = epoch_metrics.get("sampling/entropy_normalized", 1.0)
            logger.info(
                f"Epoch {epoch}: Mean performance = {epoch_metrics['task_dependency/mean_performance']:.3f}, "
                f"Sampling entropy = {sampling_entropy:.3f}"
            )

    # Get final summary
    results = simulator.get_summary()
    results["metrics_history"] = metrics_history

    # Log to wandb if available
    try:
        import wandb

        if wandb_run_name is None:
            timestamp = str(int(time.time()))
            wandb_run_name = f"task_dependency.{timestamp}"

        wandb.init(project=wandb_project, name=wandb_run_name)

        # Log configuration
        wandb.config.update(
            {
                "num_tasks": num_tasks,
                "num_epochs": num_epochs,
                "samples_per_epoch": samples_per_epoch,
                "gamma": gamma,
                "lambda_forget": lambda_forget,
                "performance_threshold": performance_threshold,
                "task_seed": task_seed,
                "ema_timescale": ema_timescale,
                "slow_timescale_factor": slow_timescale_factor,
                "exploration_bonus": exploration_bonus,
                "min_presentations_for_eviction": min_presentations_for_eviction,
                "eviction_threshold_percentile": eviction_threshold_percentile,
            }
        )

        # Log metrics for each epoch
        for epoch, metrics in enumerate(metrics_history):
            # Clean up raw score data that's not needed for logging
            epoch_metrics = metrics.copy()

            # Remove raw score arrays from metrics (they're slow to upload and not needed)
            keys_to_remove = []
            for key in list(epoch_metrics.keys()):
                if key.startswith("_learning_progress") and key.endswith("_scores"):
                    keys_to_remove.append(key)
                elif key == "_learning_progress_scores_raw":
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                epoch_metrics.pop(key, None)

            wandb.log(epoch_metrics, step=epoch)

        # Log final summary statistics (without slow histogram plots)
        final_metrics = metrics_history[-1] if metrics_history else {}
        if "_learning_progress_scores_raw" in final_metrics:
            final_scores = final_metrics["_learning_progress_scores_raw"]
            if final_scores and len(final_scores) > 0:
                wandb.log(
                    {
                        "final_summary/total_tasks": len(final_scores),
                        "final_summary/mean_score": np.mean(final_scores),
                        "final_summary/std_score": np.std(final_scores),
                    }
                )

        # Task 0 reward history charts removed for performance

        # Task 0 learning progress charts removed for performance

        # Sampling imbalance charts removed for performance

        # Log final summary
        wandb.log({"simulation_summary": results})
        wandb.finish()

        logger.info(f"✅ Results logged to wandb project: {wandb_project}")

    except ImportError:
        logger.warning("⚠️ wandb not available, skipping logging")
    except Exception as e:
        logger.warning(f"⚠️ wandb logging failed: {e}")

    logger.info(
        f"Simulation complete. Final mean performance: {results['final_mean_performance']:.3f}"
    )
    return results


class TaskDependencySimulationTool(Tool):
    """
    Tool for running task dependency simulations.

    This tool runs pure curriculum learning simulations without agents or policies,
    focusing on task dependency dynamics and learning progress analysis.
    """

    # Simulation parameters
    num_tasks: int = 10
    num_epochs: int = 100
    samples_per_epoch: int = 50
    gamma: float = 0.1
    lambda_forget: float = 0.1
    performance_threshold: float = 0.9
    task_seed: Optional[int] = None
    dt: float = 0.1
    task_noise_std: float = 1e-5
    enable_detailed_slice_logging: bool = False

    # Learning progress parameters
    ema_timescale: float = 0.05
    slow_timescale_factor: float = 1 / 5
    exploration_bonus: float = 0.1
    progress_smoothing: float = 1e-4
    use_bidirectional: bool = True
    max_memory_tasks: int = 100000  # Large enough to avoid cleanup
    max_slice_axes: int = 3
    num_active_tasks: int = 1000
    rand_task_rate: float = 0.01
    sampling_temperature: float = 1.0  # Lower = more focused sampling

    # Eviction parameters
    min_presentations_for_eviction: int = 30
    eviction_threshold_percentile: float = 0.2

    # Wandb parameters
    wandb_project: str = "task_dependency_simulator"
    wandb_run_name: Optional[str] = None

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run the task dependency simulation."""
        logger.info("Starting task dependency simulation...")

        try:
            results = simulate_task_dependencies(
                num_tasks=self.num_tasks,
                num_epochs=self.num_epochs,
                samples_per_epoch=self.samples_per_epoch,
                gamma=self.gamma,
                lambda_forget=self.lambda_forget,
                performance_threshold=self.performance_threshold,
                task_seed=self.task_seed,
                dt=self.dt,
                task_noise_std=self.task_noise_std,
                enable_detailed_slice_logging=self.enable_detailed_slice_logging,
                ema_timescale=self.ema_timescale,
                slow_timescale_factor=self.slow_timescale_factor,
                exploration_bonus=self.exploration_bonus,
                progress_smoothing=self.progress_smoothing,
                use_bidirectional=self.use_bidirectional,
                max_memory_tasks=self.max_memory_tasks,
                max_slice_axes=self.max_slice_axes,
                num_active_tasks=self.num_active_tasks,
                rand_task_rate=self.rand_task_rate,
                sampling_temperature=self.sampling_temperature,
                min_presentations_for_eviction=self.min_presentations_for_eviction,
                eviction_threshold_percentile=self.eviction_threshold_percentile,
                wandb_project=self.wandb_project,
                wandb_run_name=self.wandb_run_name,
            )

            logger.info("✅ Simulation completed successfully!")
            logger.info(
                f"Final mean performance: {results['final_mean_performance']:.3f}"
            )
            logger.info(f"Tasks above threshold: {results['tasks_above_threshold']}")

            return 0

        except Exception as e:
            logger.error(f"❌ Simulation failed: {e}")
            return 1


def simulate_large_chain(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate a large task chain (25 tasks)."""
    return TaskDependencySimulationTool(
        num_tasks=25,
        num_epochs=2000,
        samples_per_epoch=100,
        num_active_tasks=1000,  # Much smaller pool to reduce eviction overhead
        min_presentations_for_eviction=30,
        wandb_run_name=wandb_run_name,
    )


def simulate_large_chain_focused(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate a large task chain with focused sampling (low entropy ~0.5)."""
    return TaskDependencySimulationTool(
        num_tasks=25,
        num_epochs=2000,
        samples_per_epoch=100,
        num_active_tasks=30,  # Smaller pool for more focus
        exploration_bonus=0.0001,  # Much lower exploration
        ema_timescale=0.5,  # Slower adaptation
        rand_task_rate=0.001,  # Minimal randomness
        progress_smoothing=100,  # Sharper preferences
        use_bidirectional=False,  # Simpler scoring
        sampling_temperature=1e-4,  # Much more focused sampling
        min_presentations_for_eviction=300,
        wandb_run_name=wandb_run_name,
    )
