"""Curriculum Management with Automatic Hardware Detection.

This module provides a curriculum manager that automatically detects
the hardware configuration and seamlessly switches between local and distributed
modes based on the environment.
"""

import multiprocessing as mp
import os
from typing import Any, Dict, Optional

import numpy as np

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .distributed_curriculum import DistributedCurriculumConfig


class CurriculumManager:
    """Curriculum manager that automatically chooses between local and distributed modes.

    This class provides a seamless interface that automatically detects the hardware
    configuration and chooses the most appropriate curriculum management strategy:

    - **Single Process**: Uses local curriculum with learning progress
    - **Multi-Process**: Uses distributed curriculum with shared memory
    - **Multi-Machine**: Can be extended to use Redis or other distributed storage

    The interface is identical regardless of the underlying implementation.
    """

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
        aggregation_interval: int = 10,
        force_distributed: bool = False,
        force_local: bool = False,
    ):
        """Initialize the curriculum manager.

        Args:
            curriculum_config: Configuration for the curriculum
            worker_id: ID of this worker (auto-detected if None)
            num_workers: Number of workers (auto-detected if None)
            aggregation_interval: How often to perform global aggregation (distributed mode)
            force_distributed: Force distributed mode regardless of hardware
            force_local: Force local mode regardless of hardware
        """
        self.curriculum_config = curriculum_config
        self.aggregation_interval = aggregation_interval
        self.force_distributed = force_distributed
        self.force_local = force_local

        # Auto-detect hardware configuration
        self._detect_hardware_config(worker_id, num_workers)

        # Initialize the appropriate curriculum manager
        self._initialize_curriculum_manager()

    def _detect_hardware_config(self, worker_id: Optional[int], num_workers: Optional[int]):
        """Auto-detect hardware configuration and worker settings."""

        # Detect if we're in a distributed environment
        self.is_distributed = self._is_distributed_environment()

        if self.is_distributed and not self.force_local:
            # Distributed mode - need worker information
            if worker_id is None:
                self.worker_id = self._auto_detect_worker_id()
            else:
                self.worker_id = worker_id

            if num_workers is None:
                self.num_workers = self._auto_detect_num_workers()
            else:
                self.num_workers = num_workers

            print("🔗 Curriculum Manager: Distributed mode detected")
            print(f"   Worker ID: {self.worker_id}/{self.num_workers}")
            print(f"   Aggregation interval: {self.aggregation_interval}")
        else:
            # Local mode
            self.worker_id = 0
            self.num_workers = 1
            print("🏠 Curriculum Manager: Local mode detected")

    def _is_distributed_environment(self) -> bool:
        """Detect if we're in a distributed environment."""

        # Check environment variables for distributed training
        distributed_env_vars = [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",  # PyTorch DDP
            "SLURM_PROCID",
            "SLURM_NTASKS",  # SLURM
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",  # OpenMPI
            "CUDA_VISIBLE_DEVICES",  # GPU distribution
            "RAY_WORKER_ID",  # Ray
            "WANDB_RUN_ID",  # Wandb distributed runs
        ]

        for var in distributed_env_vars:
            if os.getenv(var) is not None:
                return True

        # Check if multiprocessing is being used
        if mp.current_process().name != "MainProcess":
            return True

        # Check if we're in a multi-GPU environment
        try:
            import torch

            if torch.cuda.device_count() > 1:
                return True
        except ImportError:
            pass

        return False

    def _auto_detect_worker_id(self) -> int:
        """Auto-detect worker ID from environment variables."""

        # Try various environment variables
        env_vars = [
            ("RANK", int),
            ("SLURM_PROCID", int),
            ("OMPI_COMM_WORLD_RANK", int),
            ("RAY_WORKER_ID", lambda x: int(x.split("_")[-1]) if "_" in x else int(x)),
        ]

        for var_name, converter in env_vars:
            value = os.getenv(var_name)
            if value is not None:
                try:
                    return converter(value)
                except (ValueError, TypeError):
                    continue

        # Fallback: use process ID
        return os.getpid() % 1000  # Use modulo to keep reasonable range

    def _auto_detect_num_workers(self) -> int:
        """Auto-detect number of workers from environment variables."""

        # Try various environment variables
        env_vars = [
            ("WORLD_SIZE", int),
            ("SLURM_NTASKS", int),
            ("OMPI_COMM_WORLD_SIZE", int),
        ]

        for var_name, converter in env_vars:
            value = os.getenv(var_name)
            if value is not None:
                try:
                    return converter(value)
                except (ValueError, TypeError):
                    continue

        # Fallback: check CPU count
        return mp.cpu_count()

    def _initialize_curriculum_manager(self):
        """Initialize the appropriate curriculum manager based on configuration."""

        if self.is_distributed and not self.force_local:
            # Distributed mode
            self._init_distributed_manager()
        else:
            # Local mode
            self._init_local_manager()

    def _init_distributed_manager(self):
        """Initialize distributed curriculum manager."""

        # Create distributed curriculum config
        dist_config = DistributedCurriculumConfig(
            num_tasks=self.curriculum_config.num_active_tasks,
            num_workers=self.num_workers,
            worker_id=self.worker_id,
            aggregation_interval=self.aggregation_interval,
        )

        # Create distributed manager
        self.distributed_manager = dist_config.create()

        # Create local curriculum for task generation
        self.local_curriculum = Curriculum(self.curriculum_config)

        # Track which mode we're using
        self.mode = "distributed"

        print("   Using distributed curriculum with shared memory")

    def _init_local_manager(self):
        """Initialize local curriculum manager."""

        # Create local curriculum
        self.local_curriculum = Curriculum(self.curriculum_config)

        # No distributed manager needed
        self.distributed_manager = None

        # Track which mode we're using
        self.mode = "local"

        print("   Using local curriculum with learning progress")

    def get_task(self) -> CurriculumTask:
        """Get a task from the curriculum.

        Returns:
            CurriculumTask: The selected task
        """
        if self.mode == "distributed" and self.distributed_manager is not None:
            # Use distributed sampling
            task_id = self.distributed_manager.sample_task()

            # Get the corresponding task from local curriculum
            task_list = list(self.local_curriculum._tasks.values())
            if task_id < len(task_list):
                return task_list[task_id]
            else:
                # Fallback to local curriculum if task ID is out of range
                return self.local_curriculum.get_task()
        else:
            # Use local curriculum
            return self.local_curriculum.get_task()

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance.

        Args:
            task_id: ID of the task that was completed
            score: Success rate (0.0 to 1.0) for the task
        """
        if self.mode == "distributed" and self.distributed_manager is not None:
            # Update distributed manager
            self.distributed_manager.update_task_performance(task_id, score)

            # Also update local curriculum for consistency
            self.local_curriculum.update_task_performance(task_id, score)
        else:
            # Update local curriculum
            self.local_curriculum.update_task_performance(task_id, score)

    def get_task_weights(self) -> np.ndarray:
        """Get current task weights.

        Returns:
            Array of task weights
        """
        if self.mode == "distributed" and self.distributed_manager is not None:
            return self.distributed_manager.get_task_weights()
        else:
            # Extract weights from local curriculum algorithm
            if self.local_curriculum._algorithm is not None:
                return self.local_curriculum._algorithm.weights
            else:
                # Fallback to uniform weights
                return np.ones(self.curriculum_config.num_active_tasks)

    def get_task_probabilities(self) -> np.ndarray:
        """Get current task sampling probabilities.

        Returns:
            Array of task probabilities
        """
        if self.mode == "distributed" and self.distributed_manager is not None:
            return self.distributed_manager.get_task_probabilities()
        else:
            # Extract probabilities from local curriculum algorithm
            if self.local_curriculum._algorithm is not None:
                return self.local_curriculum._algorithm.probabilities
            else:
                # Fallback to uniform probabilities
                return np.ones(self.curriculum_config.num_active_tasks) / self.curriculum_config.num_active_tasks

    def sample_task(self) -> int:
        """Sample a task ID based on current learning progress.

        Returns:
            Task ID to sample
        """
        if self.mode == "distributed" and self.distributed_manager is not None:
            return self.distributed_manager.sample_task()
        else:
            # Use local curriculum's algorithm
            if self.local_curriculum._algorithm is not None:
                return self.local_curriculum._algorithm.sample_idx()
            else:
                # Fallback to random sampling
                return np.random.randint(0, self.curriculum_config.num_active_tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the curriculum.

        Returns:
            Dictionary containing statistics from both local and distributed components
        """
        # Get local curriculum stats
        local_stats = self.local_curriculum.stats()

        # Add mode information
        stats = {"mode": self.mode, "worker_id": self.worker_id, "num_workers": self.num_workers, **local_stats}

        # Add distributed stats if applicable
        if self.mode == "distributed" and self.distributed_manager is not None:
            dist_stats = self.distributed_manager.get_stats()
            stats.update({f"distributed_{k}": v for k, v in dist_stats.items()})

        return stats

    def force_global_aggregation(self):
        """Force immediate global aggregation (distributed mode only)."""
        if self.mode == "distributed" and self.distributed_manager is not None:
            self.distributed_manager.force_global_aggregation()

    def switch_to_local_mode(self):
        """Switch to local mode (useful for debugging or single-worker scenarios)."""
        if self.mode == "distributed":
            print("🔄 Switching from distributed to local mode")
            self.mode = "local"
            # Keep the local curriculum, just stop using distributed manager

    def switch_to_distributed_mode(self, worker_id: int, num_workers: int):
        """Switch to distributed mode with specified worker configuration."""
        if self.mode == "local":
            print(f"🔄 Switching from local to distributed mode (worker {worker_id}/{num_workers})")
            self.worker_id = worker_id
            self.num_workers = num_workers
            self._init_distributed_manager()
            self.mode = "distributed"


class CurriculumManagerConfig:
    """Configuration for curriculum management."""

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        worker_id: Optional[int] = None,
        num_workers: Optional[int] = None,
        aggregation_interval: int = 10,
        force_distributed: bool = False,
        force_local: bool = False,
    ):
        """Initialize curriculum manager configuration.

        Args:
            curriculum_config: Base curriculum configuration
            worker_id: ID of this worker (auto-detected if None)
            num_workers: Number of workers (auto-detected if None)
            aggregation_interval: How often to perform global aggregation
            force_distributed: Force distributed mode
            force_local: Force local mode
        """
        self.curriculum_config = curriculum_config
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.aggregation_interval = aggregation_interval
        self.force_distributed = force_distributed
        self.force_local = force_local

    def create(self) -> CurriculumManager:
        """Create a curriculum manager instance.

        Returns:
            Configured curriculum manager
        """
        return CurriculumManager(
            curriculum_config=self.curriculum_config,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            aggregation_interval=self.aggregation_interval,
            force_distributed=self.force_distributed,
            force_local=self.force_local,
        )
