"""
Mock Dynamical System Environment for Task Dependency Experiments.

This environment bypasses mettagrid and agents entirely, implementing a pure
curriculum learning simulation where the curriculum algorithm drives task selection
and the environment simulates the underlying task dependency dynamics.
"""

import logging
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MockDynamicalSystemSimulator:
    """
    Pure curriculum learning simulator that models task dependency dynamics.

    This class simulates the learning dynamics without any agent interaction.
    The curriculum algorithm drives task selection and the simulator updates
    task performance based on the dependency chain dynamics.
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
        **kwargs,
    ):
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        self.gamma = gamma
        self.lambda_forget = lambda_forget
        self.performance_threshold = performance_threshold
        self.task_seed = task_seed or random.randint(0, 2**31 - 1)

        # Initialize task dependency chain (0 -> 1 -> 2 -> ...)
        self._build_task_chain()

        # Initialize performance tracking
        self.P = torch.full((num_tasks,), 0.01)  # Current performance
        self.current_epoch = 0
        self.epoch_sample_counts = torch.zeros(num_tasks)
        self.total_sample_counts = torch.zeros(num_tasks)

        # Task-specific noise (generated from seed)
        self._task_noise = self._generate_task_noise()

        # History for logging
        self.performance_history = [self.P.clone()]
        self.sample_history = []

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
        # Each task has a base reward mean + task-specific noise
        np.random.seed(self.task_seed)
        task_noise = np.random.normal(0.0, 0.1, size=self.num_tasks)
        return torch.tensor(task_noise, dtype=torch.float32)

    def reset_epoch(self) -> None:
        """Reset for a new epoch of curriculum learning."""
        self.current_epoch = 0
        self.epoch_sample_counts = torch.zeros(self.num_tasks)
        self.total_sample_counts = torch.zeros(self.num_tasks)
        self.P = torch.full((self.num_tasks,), 0.01)
        self.performance_history = [self.P.clone()]
        self.sample_history = []

    def sample_task(self, task_id: int) -> float:
        """
        Simulate sampling a specific task and return the reward.

        This is called by the curriculum algorithm when it selects a task.
        Returns the task-specific reward based on current performance + noise.
        """
        # Update sample count
        self.epoch_sample_counts[task_id] += 1
        self.total_sample_counts[task_id] += 1

        # Calculate reward: base + current performance + task-specific noise
        base_reward = 0.5
        task_reward = base_reward + self.P[task_id].item() + self._task_noise[task_id].item()
        reward = float(np.clip(task_reward, 0.0, 1.0))

        return reward

    def complete_epoch(self) -> Dict[str, Any]:
        """
        Complete the current epoch and update task dynamics.

        This should be called at the end of each epoch to update performance
        based on the task dependency dynamics.
        """
        # Update task performance based on dynamics
        self._update_task_dynamics()

        # Store history
        self.performance_history.append(self.P.clone())
        self.sample_history.append(self.epoch_sample_counts.clone())

        # Get metrics for logging
        info = self._get_current_metrics()

        # Advance to next epoch and reset sample counts
        self.current_epoch += 1
        self.epoch_sample_counts = torch.zeros(self.num_tasks)

        return info

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for wandb logging."""
        # Calculate additional metrics for wandb logging
        task_completion_probs = torch.sigmoid((self.P - 0.5) * 4)  # Smooth completion probability

        info = {
            "task_performances": self.P.tolist(),
            "task_completion_probs": task_completion_probs.tolist(),
            "epoch": self.current_epoch,
            "sample_counts": self.epoch_sample_counts.tolist(),
            "mean_performance": self.P.mean().item(),
            "tasks_above_threshold": (self.P >= self.performance_threshold).sum().item(),
            "max_performance": self.P.max().item(),
            "min_performance": self.P.min().item(),
            "performance_std": self.P.std().item(),
            "total_samples_this_epoch": self.epoch_sample_counts.sum().item(),
        }

        # Add individual task metrics for detailed logging
        for i in range(self.num_tasks):
            info[f"task_{i}_performance"] = self.P[i].item()
            info[f"task_{i}_completion_prob"] = task_completion_probs[i].item()
            info[f"task_{i}_samples"] = self.epoch_sample_counts[i].item()
            info[f"task_{i}_reward_noise"] = self._task_noise[i].item()

        return info

    def is_complete(self) -> bool:
        """Check if the simulation is complete."""
        return self.current_epoch >= self.num_epochs

    def _update_task_dynamics(self) -> None:
        """Update task performance based on chain dynamics."""
        current_P = self.P.clone()
        P_dot = torch.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            # Parent contribution
            parent_contribution = sum(self.epoch_sample_counts[p] for p in self.parents[i])
            total_stimulus = self.epoch_sample_counts[i] + self.gamma * parent_contribution

            # Children gate: performance gated by children performance
            children_gate = 1.0
            if self.adj[i]:  # If task i has children
                children_gate = torch.prod(torch.tensor([current_P[c] for c in self.adj[i]]))

            # Growth and forgetting
            growth = total_stimulus * children_gate * (1 - current_P[i])
            forgetting = self.lambda_forget * current_P[i]
            P_dot[i] = growth - forgetting

        # Update performance (normalized by samples per epoch)
        new_P = current_P + P_dot * (1.0 / self.samples_per_epoch)
        self.P = torch.clamp(new_P, 0, 1)

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire simulation for analysis."""
        return {
            "final_performances": self.P.tolist(),
            "performance_history": [p.tolist() for p in self.performance_history],
            "sample_history": [s.tolist() for s in self.sample_history],
            "total_samples": self.total_sample_counts.tolist(),
            "num_epochs_completed": self.current_epoch,
            "final_mean_performance": self.P.mean().item(),
            "tasks_above_threshold": (self.P >= self.performance_threshold).sum().item(),
        }


class CurriculumDrivenSimulation:
    """
    Runs a curriculum-driven simulation without agents or policies.

    This class coordinates between a curriculum algorithm and the mock simulator
    to run pure curriculum learning experiments that focus on task selection
    and dependency dynamics.
    """

    def __init__(self, simulator: MockDynamicalSystemSimulator, curriculum):
        self.simulator = simulator
        self.curriculum = curriculum
        self.metrics_history = []

    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        logger.info(f"Starting curriculum simulation for {self.simulator.num_epochs} epochs")

        # Reset simulator
        self.simulator.reset_epoch()

        for epoch in range(self.simulator.num_epochs):
            # Simulate curriculum-driven task sampling for this epoch
            for _ in range(self.simulator.samples_per_epoch):
                # Get task from curriculum (this would normally happen in environment step)
                task = self.curriculum.get_task()
                task_id = task._task_id % self.simulator.num_tasks  # Ensure valid task ID

                # Sample the task and get reward
                reward = self.simulator.sample_task(task_id)

                # Complete task with reward (update curriculum algorithm)
                task.complete(reward)
                self.curriculum.update_task_performance(task_id, reward)

            # Complete epoch and get metrics
            epoch_metrics = self.simulator.complete_epoch()
            epoch_metrics["curriculum_stats"] = self.curriculum.stats()
            self.metrics_history.append(epoch_metrics)

            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Mean performance = {epoch_metrics['mean_performance']:.3f}")

        # Get final summary
        summary = self.simulator.get_simulation_summary()
        summary["metrics_history"] = self.metrics_history

        logger.info(f"Simulation complete. Final mean performance: {summary['final_mean_performance']:.3f}")

        return summary
