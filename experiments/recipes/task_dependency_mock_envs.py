"""
Task Dependency Mock Environment Recipe.

This recipe creates a mock environment that simulates task dependency learning
without using mettagrid. It implements a chain-based task structure where
tasks have dependencies and learning follows dynamical system equations.
"""

import os
import time
from typing import Any, Dict, Optional

from metta.cogworks.curriculum import Curriculum, CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.mock_dynamical_env import (
    CurriculumDrivenSimulation,
    MockDynamicalSystemSimulator,
)
from mettagrid.config.mettagrid_config import MettaGridConfig
from pydantic import BaseModel


class MockEnvironmentConfig(BaseModel):
    """Configuration for the mock dynamical system environment."""

    num_tasks: int = 10
    num_epochs: int = 100
    samples_per_epoch: int = 50
    gamma: float = 0.1  # Parent contribution factor
    lambda_forget: float = 0.1  # Forgetting rate
    performance_threshold: float = 0.9
    task_seed: Optional[int] = None


class MockTaskGenerator(TaskGenerator):
    """Task generator for mock environment that creates tasks with different seeds."""

    class Config(TaskGeneratorConfig["MockTaskGenerator"]):
        mock_env_config: MockEnvironmentConfig = MockEnvironmentConfig()

    def __init__(self, config: "MockTaskGenerator.Config"):
        super().__init__(config)
        self.mock_config = config.mock_env_config

    def _generate_task(self, task_id: int, rng) -> MettaGridConfig:
        """Generate a MettaGridConfig with mock environment parameters embedded."""
        # Create a minimal valid mettagrid config (required by curriculum interface)
        mock_mg_config = MettaGridConfig(
            label=f"mock_task_{task_id}",
        )

        # Store mock environment parameters on the config
        mock_mg_config._mock_env_params = {
            "task_seed": task_id,
            **self.mock_config.model_dump(),
        }

        return mock_mg_config


def _default_run_name() -> str:
    """Generate a default run name for task dependency experiments."""
    timestamp = str(int(time.time()))
    try:
        user = os.getenv("USER", "unknown")
        return f"task_dependency_mock.{user}.{timestamp}"
    except Exception:
        return f"task_dependency_mock.{timestamp}"


def make_mock_env_config(
    num_tasks: int = 10,
    num_epochs: int = 100,
    samples_per_epoch: int = 50,
    gamma: float = 0.1,
    lambda_forget: float = 0.1,
    performance_threshold: float = 0.9,
) -> MockEnvironmentConfig:
    """Create a mock environment configuration."""
    return MockEnvironmentConfig(
        num_tasks=num_tasks,
        num_epochs=num_epochs,
        samples_per_epoch=samples_per_epoch,
        gamma=gamma,
        lambda_forget=lambda_forget,
        performance_threshold=performance_threshold,
    )


def make_curriculum(
    mock_env_config: Optional[MockEnvironmentConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[LearningProgressConfig] = None,
) -> CurriculumConfig:
    """Create curriculum configuration for mock environment."""
    mock_env_config = mock_env_config or make_mock_env_config()

    # Create task generator config
    task_gen_config = MockTaskGenerator.Config(
        mock_env_config=mock_env_config,
    )

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=3,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
            num_active_tasks=min(16, mock_env_config.num_tasks),
            rand_task_rate=0.25,
        )

    return CurriculumConfig(
        task_generator=task_gen_config,
        algorithm_config=algorithm_config,
        num_active_tasks=min(16, mock_env_config.num_tasks),
    )


def simulate(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    mock_env_config: Optional[MockEnvironmentConfig] = None,
    wandb_project: str = "task_dependency_mock",
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a curriculum-driven simulation of task dependency dynamics."""
    mock_env_config = mock_env_config or make_mock_env_config()

    # Create curriculum
    curriculum_config = curriculum or make_curriculum(
        mock_env_config=mock_env_config,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )
    curriculum_instance = Curriculum(curriculum_config)

    # Create simulator
    simulator = MockDynamicalSystemSimulator(**mock_env_config.model_dump())

    # Create and run simulation
    simulation = CurriculumDrivenSimulation(simulator, curriculum_instance)
    results = simulation.run_simulation()

    # Log to wandb if available
    try:
        import wandb

        if wandb_run_name is None:
            wandb_run_name = _default_run_name()

        # Initialize wandb run
        wandb.init(project=wandb_project, name=wandb_run_name)

        # Log metrics for each epoch
        for epoch, metrics in enumerate(results["metrics_history"]):
            wandb.log(metrics, step=epoch)

        # Log final summary
        wandb.log({"final_summary": results}, step=len(results["metrics_history"]))

        wandb.finish()
        print(f"✅ Results logged to wandb project: {wandb_project}")

    except ImportError:
        print("⚠️ wandb not available, skipping logging")
    except Exception as e:
        print(f"⚠️ wandb logging failed: {e}")

    return results


# Convenience functions for different experiment configurations


def simulate_small_chain(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Simulate a small task chain (5 tasks)."""
    mock_config = make_mock_env_config(
        num_tasks=5,
        num_epochs=50,
        samples_per_epoch=25,
    )
    return simulate(mock_env_config=mock_config, wandb_run_name=wandb_run_name)


def simulate_large_chain(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Simulate a large task chain (20 tasks)."""
    mock_config = make_mock_env_config(
        num_tasks=20,
        num_epochs=200,
        samples_per_epoch=100,
    )
    return simulate(mock_env_config=mock_config, wandb_run_name=wandb_run_name)


def simulate_high_gamma(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Simulate with high parent contribution (gamma=0.3)."""
    mock_config = make_mock_env_config(
        gamma=0.3,  # High parent contribution
        lambda_forget=0.05,  # Lower forgetting
    )
    return simulate(mock_env_config=mock_config, wandb_run_name=wandb_run_name)


def simulate_high_forgetting(wandb_run_name: Optional[str] = None) -> Dict[str, Any]:
    """Simulate with high forgetting rate."""
    mock_config = make_mock_env_config(
        gamma=0.05,  # Low parent contribution
        lambda_forget=0.2,  # High forgetting
    )
    return simulate(mock_env_config=mock_config, wandb_run_name=wandb_run_name)
