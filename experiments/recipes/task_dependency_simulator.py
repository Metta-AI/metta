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

        # History for analysis
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
        np.random.seed(self.task_seed)
        task_noise = np.random.normal(0.0, 0.1, size=self.num_tasks)
        return torch.tensor(task_noise, dtype=torch.float32)

    def reset(self) -> None:
        """Reset simulator state."""
        self.current_epoch = 0
        self.epoch_sample_counts = torch.zeros(self.num_tasks)
        self.total_sample_counts = torch.zeros(self.num_tasks)
        self.P = torch.full((self.num_tasks,), 0.01)
        self.performance_history = [self.P.clone()]
        self.sample_history = []

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

        return reward

    def complete_epoch(self) -> Dict[str, Any]:
        """Complete epoch and update task dynamics."""
        # Update task performance based on dynamics
        self._update_task_dynamics()

        # Store history
        self.performance_history.append(self.P.clone())
        self.sample_history.append(self.epoch_sample_counts.clone())

        # Get metrics for logging
        metrics = self._get_epoch_metrics()

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

        # Update performance (normalized by samples per epoch)
        new_P = current_P + P_dot * (1.0 / self.samples_per_epoch)
        self.P = torch.clamp(new_P, 0, 1)

    def _get_epoch_metrics(self) -> Dict[str, Any]:
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

        # Add individual task metrics (limit to first 10 tasks for wandb)
        for i in range(min(self.num_tasks, 10)):
            metrics[f"task_dependency/task_{i}_performance"] = self.P[i].item()
            metrics[f"task_dependency/task_{i}_completion_prob"] = (
                task_completion_probs[i].item()
            )
            metrics[f"task_dependency/task_{i}_samples"] = self.epoch_sample_counts[
                i
            ].item()

        return metrics

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
) -> CurriculumConfig:
    """Create curriculum configuration for task dependency simulation."""
    task_gen_config = MockTaskGenerator.Config()

    algorithm_config = LearningProgressConfig(
        use_bidirectional=True,
        ema_timescale=0.001,
        exploration_bonus=0.1,
        max_memory_tasks=1000,
        max_slice_axes=3,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        num_active_tasks=min(16, num_tasks),
        rand_task_rate=0.25,
    )

    return CurriculumConfig(
        task_generator=task_gen_config,
        algorithm_config=algorithm_config,
        num_active_tasks=min(16, num_tasks),
    )


def simulate_task_dependencies(
    num_tasks: int = 10,
    num_epochs: int = 100,
    samples_per_epoch: int = 50,
    gamma: float = 0.1,
    lambda_forget: float = 0.1,
    performance_threshold: float = 0.9,
    task_seed: Optional[int] = None,
    enable_detailed_slice_logging: bool = False,
    wandb_project: str = "task_dependency_simulator",
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
    )

    # Create curriculum
    curriculum_config = create_curriculum(
        num_tasks=num_tasks,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
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
            curriculum.update_task_performance(task_id, reward)

        # Complete epoch and collect metrics
        epoch_metrics = simulator.complete_epoch()
        epoch_metrics.update(curriculum.stats())
        metrics_history.append(epoch_metrics)

        # Log progress
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Mean performance = {epoch_metrics['task_dependency/mean_performance']:.3f}"
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
            }
        )

        # Log metrics for each epoch
        for epoch, metrics in enumerate(metrics_history):
            wandb.log(metrics, step=epoch)

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


# Convenience functions for programmatic use (return dictionaries)


def run_small_chain_simulation(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Programmatically run a small task chain simulation (5 tasks)."""
    return simulate_task_dependencies(
        num_tasks=5,
        num_epochs=500,
        samples_per_epoch=25,
        wandb_run_name=wandb_run_name,
    )


def run_large_chain_simulation(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Programmatically run a large task chain simulation (20 tasks)."""
    return simulate_task_dependencies(
        num_tasks=20,
        num_epochs=2000,
        samples_per_epoch=100,
        wandb_run_name=wandb_run_name,
    )


def run_high_gamma_simulation(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Programmatically run simulation with high parent contribution (gamma=0.3)."""
    return simulate_task_dependencies(
        gamma=0.3,  # High parent contribution
        lambda_forget=0.05,  # Lower forgetting
        wandb_run_name=wandb_run_name,
    )


def run_high_forgetting_simulation(
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Programmatically run simulation with high forgetting rate."""
    return simulate_task_dependencies(
        gamma=0.05,  # Low parent contribution
        lambda_forget=0.2,  # High forgetting
        wandb_run_name=wandb_run_name,
    )


# Tool implementations for the recipe system


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
    enable_detailed_slice_logging: bool = False

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
                enable_detailed_slice_logging=self.enable_detailed_slice_logging,
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


# Recipe functions that return Tool instances


def simulate_small_chain(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate a small task chain (5 tasks)."""
    return TaskDependencySimulationTool(
        num_tasks=5,
        num_epochs=500,
        samples_per_epoch=25,
        wandb_run_name=wandb_run_name,
    )


def simulate_large_chain(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate a large task chain (20 tasks)."""
    return TaskDependencySimulationTool(
        num_tasks=20,
        num_epochs=2000,
        samples_per_epoch=100,
        wandb_run_name=wandb_run_name,
    )


def simulate_high_gamma(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate with high parent contribution (gamma=0.3)."""
    return TaskDependencySimulationTool(
        gamma=0.3,  # High parent contribution
        lambda_forget=0.05,  # Lower forgetting
        wandb_run_name=wandb_run_name,
    )


def simulate_high_forgetting(
    wandb_run_name: Optional[str] = None,
) -> TaskDependencySimulationTool:
    """Simulate with high forgetting rate."""
    return TaskDependencySimulationTool(
        gamma=0.05,  # Low parent contribution
        lambda_forget=0.2,  # High forgetting
        wandb_run_name=wandb_run_name,
    )
