"""
Curriculum builder utilities for notebooks.

This module provides a convenient API for building curriculum configurations
programmatically in notebooks, without needing to write YAML files manually.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml
from omegaconf import DictConfig, OmegaConf


class TaskBuilder:
    """Builder for creating individual tasks with environment configurations."""

    def __init__(self, env_cfg: DictConfig):
        """
        Create a task builder with a base environment configuration.

        Args:
            env_cfg: The environment configuration for this task
        """
        self.env_cfg = env_cfg
        self._overrides: Dict[str, Any] = {}

    def override(self, key: str, value: Any) -> "TaskBuilder":
        """
        Override a specific parameter in the environment config.

        Args:
            key: Dot-separated path to the parameter (e.g., "game.num_agents")
            value: New value for the parameter

        Returns:
            Self for method chaining
        """
        self._overrides[key] = value
        return self

    def build(self) -> DictConfig:
        """
        Build the final task configuration.

        Returns:
            The task configuration with all overrides applied
        """
        # Convert to dict if it's a Pydantic model
        if hasattr(self.env_cfg, "model_dump"):
            # Pydantic v2
            config_dict = self.env_cfg.model_dump()
        elif hasattr(self.env_cfg, "dict"):
            # Pydantic v1
            config_dict = self.env_cfg.dict()
        else:
            # Already a dict or DictConfig
            config_dict = self.env_cfg

        # Create OmegaConf from dict
        task_cfg = OmegaConf.create(config_dict)

        # Apply overrides
        from omegaconf import open_dict

        with open_dict(task_cfg):
            for key, value in self._overrides.items():
                keys = key.split(".")
                current = task_cfg
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value

        return task_cfg


class CurriculumBuilder:
    """Builder for creating curriculum configurations."""

    def __init__(self):
        self.tasks: Dict[str, Union[str, DictConfig]] = {}
        self.task_weights: Dict[str, float] = {}
        self.env_overrides: Dict[str, Any] = {}

    def add_task(
        self,
        task_id: str,
        task_config: Union[str, DictConfig, TaskBuilder],
        weight: float = 1.0,
    ) -> "CurriculumBuilder":
        """
        Add a task to the curriculum.

        Args:
            task_id: Unique identifier for the task
            task_config: Task configuration (path to YAML, DictConfig, or TaskBuilder)
            weight: Weight for task selection (default 1.0)

        Returns:
            Self for method chaining
        """
        if isinstance(task_config, TaskBuilder):
            task_config = task_config.build()

        self.tasks[task_id] = task_config
        self.task_weights[task_id] = weight
        return self

    def add_env_override(self, key: str, value: Any) -> "CurriculumBuilder":
        """
        Add an environment override that applies to all tasks.

        Args:
            key: Dot-separated path to the parameter
            value: New value for the parameter

        Returns:
            Self for method chaining
        """
        self.env_overrides[key] = value
        return self

    def build_random(self) -> Dict[str, Any]:
        """
        Build a RandomCurriculum configuration.

        Returns:
            Dictionary representing the curriculum config
        """
        config = {
            "_target_": "metta.mettagrid.curriculum.random.RandomCurriculum",
            "tasks": {task_id: weight for task_id, weight in self.task_weights.items()},
        }

        if self.env_overrides:
            config["env_overrides"] = self.env_overrides

        return config

    def build_prioritize_regressed(self) -> Dict[str, Any]:
        """
        Build a PrioritizeRegressedCurriculum configuration.

        Returns:
            Dictionary representing the curriculum config
        """
        config = {
            "_target_": "metta.mettagrid.curriculum.prioritize_regressed.PrioritizeRegressedCurriculum",
            "tasks": {task_id: weight for task_id, weight in self.task_weights.items()},
        }

        if self.env_overrides:
            config["env_overrides"] = self.env_overrides

        return config

    def save_tasks_as_configs(self, base_dir: Path) -> Dict[str, str]:
        """
        Save task configurations as YAML files and return path mappings.

        Args:
            base_dir: Directory to save task configs

        Returns:
            Mapping from task_id to config file path
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        task_paths = {}
        for task_id, task_config in self.tasks.items():
            if isinstance(task_config, str):
                # Already a path
                task_paths[task_id] = task_config
            else:
                # Save DictConfig as YAML
                config_path = base_dir / f"{task_id}.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(OmegaConf.to_container(task_config), f, indent=2)
                task_paths[task_id] = str(config_path)

        return task_paths


class BucketBuilder:
    """Builder for creating bucketed curriculum configurations."""

    def __init__(self, base_task_config: Union[str, DictConfig, TaskBuilder]):
        """
        Create a bucket builder with a base task configuration.

        Args:
            base_task_config: Base task config to generate variations from
        """
        if isinstance(base_task_config, TaskBuilder):
            base_task_config = base_task_config.build()

        self.base_task_config = base_task_config
        self.buckets: Dict[str, Any] = {}
        self.env_overrides: Dict[str, Any] = {}
        self.default_bins: int = 1

    def add_bucket_range(
        self, key: str, min_val: float, max_val: float, bins: int = None
    ) -> "BucketBuilder":
        """
        Add a bucketed parameter with a numeric range.

        Args:
            key: Dot-separated path to the parameter
            min_val: Minimum value
            max_val: Maximum value
            bins: Number of bins (uses default_bins if not specified)

        Returns:
            Self for method chaining
        """
        bucket_spec = {"range": [min_val, max_val]}
        if bins is not None:
            bucket_spec["bins"] = bins

        self.buckets[key] = bucket_spec
        return self

    def add_bucket_values(self, key: str, values: List[Any]) -> "BucketBuilder":
        """
        Add a bucketed parameter with discrete values.

        Args:
            key: Dot-separated path to the parameter
            values: List of discrete values to use

        Returns:
            Self for method chaining
        """
        self.buckets[key] = values
        return self

    def set_default_bins(self, bins: int) -> "BucketBuilder":
        """
        Set the default number of bins for range buckets.

        Args:
            bins: Default number of bins

        Returns:
            Self for method chaining
        """
        self.default_bins = bins
        return self

    def add_env_override(self, key: str, value: Any) -> "BucketBuilder":
        """
        Add an environment override that applies to all generated tasks.

        Args:
            key: Dot-separated path to the parameter
            value: New value for the parameter

        Returns:
            Self for method chaining
        """
        self.env_overrides[key] = value
        return self

    def build(self, save_base_task: bool = True) -> Dict[str, Any]:
        """
        Build a BucketedCurriculum configuration.

        Args:
            save_base_task: Whether to save the base task config to a temp file

        Returns:
            Dictionary representing the bucketed curriculum config
        """
        config = {
            "_target_": "metta.mettagrid.curriculum.bucketed.BucketedCurriculum",
            "buckets": self.buckets,
            "default_bins": self.default_bins,
        }

        if isinstance(self.base_task_config, str):
            # It's already a path
            config["env_cfg_template_path"] = self.base_task_config
        else:
            if save_base_task:
                # Convert to dict if needed
                if hasattr(self.base_task_config, "model_dump"):
                    config_dict = self.base_task_config.model_dump()
                elif hasattr(self.base_task_config, "dict"):
                    config_dict = self.base_task_config.dict()
                else:
                    config_dict = (
                        OmegaConf.to_container(self.base_task_config)
                        if hasattr(self.base_task_config, "__dict__")
                        else self.base_task_config
                    )

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as f:
                    yaml.dump(config_dict, f, indent=2)
                    config["env_cfg_template_path"] = f.name
            else:
                # Use the config directly
                config["env_cfg_template"] = self.base_task_config

        if self.env_overrides:
            config["env_overrides"] = self.env_overrides

        return config


def save_curriculum_config(config: Dict[str, Any], path: str) -> str:
    """
    Save a curriculum configuration to a YAML file.

    Args:
        config: The curriculum configuration dictionary
        path: Path where to save the config

    Returns:
        The actual path where the config was saved
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, indent=2)

    return str(config_path)


# Convenience functions for common patterns


def simple_curriculum(*task_configs, weights: List[float] = None) -> CurriculumBuilder:
    """
    Create a simple curriculum from multiple task configurations.

    Args:
        *task_configs: Task configurations (DictConfig, TaskBuilder, or paths)
        weights: Optional weights for each task

    Returns:
        CurriculumBuilder with the tasks added
    """
    builder = CurriculumBuilder()

    if weights is None:
        weights = [1.0] * len(task_configs)
    elif len(weights) != len(task_configs):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of tasks ({len(task_configs)})"
        )

    for i, (task_config, weight) in enumerate(zip(task_configs, weights)):
        task_id = f"task_{i}"
        builder.add_task(task_id, task_config, weight)

    return builder


def bucketed_curriculum(
    base_task: Union[str, DictConfig, TaskBuilder],
) -> BucketBuilder:
    """
    Create a bucketed curriculum from a base task configuration.

    Args:
        base_task: Base task configuration

    Returns:
        BucketBuilder for the curriculum
    """
    return BucketBuilder(base_task)
