"""
New curriculum system architecture.

This module implements a redesigned curriculum system with better separation of concerns:
- TaskSet: Generates tasks deterministically from seeds, handles composition and sampling
- Curriculum: Manages task selection strategy and learning progress
- Task: Simple wrapper around a deterministic EnvConfig
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Union, Optional
from abc import ABC, abstractmethod
import copy
from omegaconf import OmegaConf
from metta.mettagrid.mettagrid_config import EnvConfig


class Task:
    """Simple wrapper around a deterministic environment configuration."""

    def __init__(self, env_config: EnvConfig, task_id: str = None):
        """
        Create a task with a deterministic environment configuration.

        Args:
            env_config: The environment configuration
            task_id: Optional identifier for the task
        """
        self.env_config = env_config
        self.task_id = task_id or f"task_{id(env_config)}"

    def get_env_config(self) -> EnvConfig:
        """Get the environment configuration for this task."""
        return self.env_config


class TaskSet(ABC):
    """
    Base class for generating tasks on-demand from a seed.

    TaskSets always use a deterministic RNG initialized with a seed to ensure
    reproducible task generation.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the TaskSet with a seed.

        Args:
            seed: Random seed for deterministic task generation
        """
        self.seed = seed

    @abstractmethod
    def get_task(self) -> EnvConfig:
        """
        Generate a task configuration using the internal RNG.

        Returns:
            Environment configuration for the generated task
        """
        pass

    def _init_rng(self) -> random.Random:
        """Initialize and return a Random instance with the TaskSet's seed."""
        rng = random.Random()
        rng.seed(self.seed)
        return rng


class WeightedTaskSet(TaskSet):
    """
    TaskSet that samples from a weighted list of EnvConfigs or other TaskSets.
    """

    def __init__(
        self,
        items: List[tuple[Union[EnvConfig, TaskSet], float]],
        overrides: Optional[Union[Dict, List[str]]] = None,
        seed: int = 42,
    ):
        """
        Create a weighted TaskSet.

        Args:
            items: List of (item, weight) tuples where item is EnvConfig or TaskSet
            overrides: Optional overrides to apply to returned configs
            seed: Random seed
        """
        super().__init__(seed)
        self.items = items
        self.overrides = self._parse_overrides(overrides) if overrides else {}

        # Validate weights
        if not items:
            raise ValueError("Items list cannot be empty")
        if any(weight <= 0 for _, weight in items):
            raise ValueError("All weights must be positive")

    def _parse_overrides(self, overrides: Union[Dict, List[str]]) -> Dict:
        """Parse overrides from either dict or list of 'key: value' strings."""
        if isinstance(overrides, dict):
            return overrides

        parsed = {}
        for override in overrides:
            if ":" not in override:
                raise ValueError(
                    f"Override must be in format 'key: value', got: {override}"
                )
            key, value = override.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Try to parse as number, boolean, or keep as string
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                # else keep as string

            parsed[key] = value
        return parsed

    def get_task(self) -> EnvConfig:
        """Sample an item by weight and return the task configuration."""
        rng = self._init_rng()

        # Sample by weight using choices (more robust)
        items = [item for item, _ in self.items]
        weights = [weight for _, weight in self.items]
        selected_item = rng.choices(items, weights=weights, k=1)[0]

        # Get the config
        if isinstance(selected_item, EnvConfig):
            config = selected_item
        else:  # It's a TaskSet
            config = selected_item.get_task()

        # Apply overrides
        if self.overrides:
            config = self._apply_overrides(config, self.overrides)

        return config

    def _apply_overrides(self, config: EnvConfig, overrides: Dict) -> EnvConfig:
        """Apply overrides to an environment configuration."""
        # Convert to dict for manipulation
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "dict"):
            config_dict = config.dict()
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

        # Apply overrides using dot notation
        config_omega = OmegaConf.create(config_dict)

        from omegaconf import open_dict

        with open_dict(config_omega):
            for key, value in overrides.items():
                keys = key.split(".")
                current = config_omega
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value

        # Convert back to EnvConfig
        updated_dict = OmegaConf.to_container(config_omega)
        return EnvConfig(**updated_dict)


class BucketedTaskSet(TaskSet):
    """
    TaskSet that generates tasks by sampling from parameter buckets.
    """

    def __init__(
        self,
        base_config: EnvConfig,
        buckets: Dict[str, Any],
        overrides: Optional[Union[Dict, List[str]]] = None,
        seed: int = 42,
    ):
        """
        Create a bucketed TaskSet.

        Args:
            base_config: Base environment configuration to modify
            buckets: Dict mapping parameter paths to bucket specs
            overrides: Optional additional overrides
            seed: Random seed
        """
        super().__init__(seed)
        self.base_config = base_config
        self.buckets = buckets
        self.overrides = (
            WeightedTaskSet._parse_overrides(None, overrides) if overrides else {}
        )

    def get_task(self) -> EnvConfig:
        """Generate a task by sampling from the buckets."""
        rng = self._init_rng()

        # Sample from each bucket
        sampled_overrides = {}
        for param_path, bucket_spec in self.buckets.items():
            sampled_overrides[param_path] = self._sample_bucket(bucket_spec, rng)

        # Combine with base overrides
        all_overrides = {**self.overrides, **sampled_overrides}

        # Apply to base config
        weighted_task_set = WeightedTaskSet(
            [(self.base_config, 1.0)], all_overrides, seed=self.seed
        )
        return weighted_task_set.get_task()

    def _sample_bucket(self, bucket_spec: Any, rng: random.Random) -> Any:
        """Sample a value from a bucket specification."""
        if isinstance(bucket_spec, list):
            # Discrete values
            return rng.choice(bucket_spec)
        elif isinstance(bucket_spec, dict) and "range" in bucket_spec:
            # Range specification
            min_val, max_val = bucket_spec["range"]
            if isinstance(min_val, int) and isinstance(max_val, int):
                return rng.randint(min_val, max_val)
            else:
                return rng.uniform(min_val, max_val)
        else:
            # Single value
            return bucket_spec


class Curriculum(ABC):
    """
    Base class for curriculum strategies that decide which tasks to return.

    Curricula take a TaskSet and implement different strategies for task selection.
    """

    def __init__(self, task_set: TaskSet):
        """
        Initialize curriculum with a TaskSet.

        Args:
            task_set: The TaskSet to generate tasks from
        """
        self.task_set = task_set

    @abstractmethod
    def get_task(self) -> Task:
        """
        Get the next task according to the curriculum strategy.

        Returns:
            A Task wrapping an environment configuration
        """
        pass


class RandomCurriculum(Curriculum):
    """
    Curriculum that generates random tasks by using random seeds.
    """

    def __init__(self, task_set: TaskSet, seed: int = 42):
        """
        Initialize random curriculum.

        Args:
            task_set: TaskSet to sample from
            seed: Seed for generating random seeds (meta-seed)
        """
        super().__init__(task_set)
        self.rng = random.Random(seed)

    def get_task(self) -> Task:
        """Generate a task using a random seed."""
        # Generate a new seed and create a new TaskSet with that seed
        task_seed = self.rng.randint(0, 2**31 - 1)

        # Create a copy of the TaskSet with the new seed
        task_set_copy = copy.deepcopy(self.task_set)
        task_set_copy.seed = task_seed

        # Generate the task
        env_config = task_set_copy.get_task()
        return Task(env_config, f"random_task_{task_seed}")


class LearningProgressCurriculum(Curriculum):
    """
    Curriculum that generates N tasks and uses learning progress for selection.
    """

    def __init__(self, task_set: TaskSet, num_tasks: int = 10, seed: int = 42):
        """
        Initialize learning progress curriculum.

        Args:
            task_set: TaskSet to sample from
            num_tasks: Number of tasks to pre-generate
            seed: Seed for task generation
        """
        super().__init__(task_set)
        self.num_tasks = num_tasks
        self.rng = random.Random(seed)

        # Pre-generate tasks
        self.tasks = []
        for i in range(num_tasks):
            task_seed = self.rng.randint(0, 2**31 - 1)
            task_set_copy = copy.deepcopy(self.task_set)
            task_set_copy.seed = task_seed
            env_config = task_set_copy.get_task()
            self.tasks.append(Task(env_config, f"lp_task_{i}"))

        # Track performance for learning progress
        self.task_scores = {task.task_id: [] for task in self.tasks}
        self.task_counts = {task.task_id: 0 for task in self.tasks}

    def get_task(self) -> Task:
        """Select task based on learning progress (simplified version)."""
        # For now, just select randomly from pre-generated tasks
        # In a full implementation, this would use learning progress metrics
        return self.rng.choice(self.tasks)

    def update_task_score(self, task_id: str, score: float):
        """Update the score for a task (for learning progress calculation)."""
        if task_id in self.task_scores:
            self.task_scores[task_id].append(score)
            self.task_counts[task_id] += 1
