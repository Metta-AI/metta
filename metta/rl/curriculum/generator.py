"""Task generators that create environment configurations from task IDs."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _convert_numpy_to_python(value: Any) -> Any:
    """Convert numpy types to native Python types for OmegaConf compatibility."""
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.str_):
        return str(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


class TaskGenerator(ABC):
    """Base class for generating task configurations from task IDs."""

    def __init__(self, base_config: DictConfig):
        """
        Initialize the task generator.

        Args:
            base_config: Base environment configuration to modify
        """
        self.base_config = base_config

    def generate(self, task_id: int) -> DictConfig:
        """
        Generate env config for given task_id.

        Args:
            task_id: Unique task identifier used as RNG seed

        Returns:
            Complete environment configuration
        """
        # Set RNG seed based on task_id for reproducibility
        rng = np.random.RandomState(task_id)

        # Create a copy of base config
        env_cfg = OmegaConf.create(OmegaConf.to_container(self.base_config, resolve=True))

        # Apply task-specific modifications
        self._apply_task_params(env_cfg, rng, task_id)

        # Set the task ID in the config for tracking
        if "task" not in env_cfg:
            env_cfg.task = {}
        env_cfg.task.id = task_id

        return env_cfg

    @abstractmethod
    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Apply task-specific parameter modifications."""
        pass


class BucketedTaskGenerator(TaskGenerator):
    """Generate tasks by sampling from parameter buckets (similar to BucketedCurriculum)."""

    def __init__(self, base_config: DictConfig, buckets: Dict[str, Any], default_bins: int = 1):
        """
        Initialize the bucketed task generator.

        Args:
            base_config: Base environment configuration
            buckets: Parameter buckets to sample from
            default_bins: Default number of bins for range parameters
        """
        super().__init__(base_config)
        self.buckets = self._expand_buckets(buckets, default_bins)
        logger.info(f"BucketedTaskGenerator initialized with {len(self.buckets)} parameters")

    def _expand_buckets(self, buckets: Dict[str, Any], default_bins: int = 1) -> Dict[str, Any]:
        """
        Expand bucket specifications into concrete values.

        Handles:
        - Range specifications: {"range": [min, max], "bins": n}
        - List specifications: [value1, value2, ...]
        """
        expanded = {}

        for parameter, bucket_spec in buckets.items():
            if isinstance(bucket_spec, dict) and "range" in bucket_spec:
                # Handle range specification
                lo, hi = bucket_spec["range"]
                n = int(bucket_spec.get("bins", default_bins))
                step = (hi - lo) / n
                want_int = isinstance(lo, int) and isinstance(hi, int)

                binned_ranges = []
                for i in range(n):
                    lo_i, hi_i = lo + i * step, lo + (i + 1) * step
                    binned_ranges.append({"range": (lo_i, hi_i), "want_int": want_int})

                expanded[parameter] = binned_ranges
            else:
                # Handle list specification
                if not isinstance(bucket_spec, (list, ListConfig)):
                    raise ValueError(
                        f"Bucket spec for {parameter} must be {{range: [lo, hi]}} or list. Got: {bucket_spec}"
                    )
                expanded[parameter] = list(bucket_spec)

        return expanded

    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Apply bucketed parameter sampling."""
        for param_path, values in self.buckets.items():
            # Sample a value
            if isinstance(values[0], dict) and "range" in values[0]:
                # Sample from range
                bucket = rng.choice(values)
                lo, hi = bucket["range"]
                value = rng.uniform(lo, hi)
                if bucket.get("want_int", False):
                    value = int(value)
            else:
                # Sample discrete value
                value = rng.choice(values)

            # Set in config using dot notation
            self._set_nested_value(env_cfg, param_path, value)

    def _set_nested_value(self, cfg: DictConfig, path: str, value: Any):
        """Set a value in nested config using dot notation."""
        parts = path.split(".")
        current = cfg

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value, converting numpy types to Python types
        current[parts[-1]] = _convert_numpy_to_python(value)


class RandomTaskGenerator(TaskGenerator):
    """Generate tasks by sampling from predefined task types (similar to RandomCurriculum)."""

    def __init__(self, task_configs: Dict[str, DictConfig], weights: Optional[Dict[str, float]] = None):
        """
        Initialize the random task generator.

        Args:
            task_configs: Dictionary of task type name -> configuration
            weights: Optional weights for each task type (uniform if not provided)
        """
        # Use the first task config as base
        super().__init__(next(iter(task_configs.values())))

        self.task_configs = task_configs
        self.weights = weights or {k: 1.0 for k in task_configs}

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"RandomTaskGenerator initialized with {len(task_configs)} task types")

    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Select and apply a random task type configuration."""
        # Select a task type based on weights
        task_types = list(self.task_configs.keys())
        probs = [self.weights[t] for t in task_types]

        selected_type = rng.choice(task_types, p=probs)
        # Convert numpy string to native Python string
        selected_type = _convert_numpy_to_python(selected_type)

        # Merge with selected task config
        selected_config = self.task_configs[selected_type]
        # Merge in-place by updating each key-value pair
        for key, value in selected_config.items():
            if isinstance(value, DictConfig) and key in env_cfg and isinstance(env_cfg[key], DictConfig):
                # Recursively merge nested configs
                env_cfg[key] = OmegaConf.merge(env_cfg[key], value)
            else:
                env_cfg[key] = value

        # Store task type for tracking
        if "task" not in env_cfg:
            env_cfg.task = {}
        env_cfg.task.type = selected_type


class SampledTaskGenerator(TaskGenerator):
    """Generate tasks by sampling specific parameter values (used by BucketedCurriculum)."""

    def __init__(self, base_config: DictConfig, sampling_parameters: Dict[str, Any]):
        """
        Initialize the sampled task generator.

        Args:
            base_config: Base environment configuration
            sampling_parameters: Dictionary of parameter paths to their sampling specs
        """
        super().__init__(base_config)
        self.sampling_parameters = sampling_parameters

    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Apply sampled parameters to the configuration."""
        for param_path, value_spec in self.sampling_parameters.items():
            if isinstance(value_spec, dict) and "range" in value_spec:
                # Sample from range
                lo, hi = value_spec["range"]
                value = rng.uniform(lo, hi)
                if value_spec.get("want_int", False):
                    value = int(value)
            else:
                # Use value directly
                value = value_spec

            # Set in config
            self._set_nested_value(env_cfg, param_path, value)

    def _set_nested_value(self, cfg: DictConfig, path: str, value: Any):
        """Set a value in nested config using dot notation."""
        parts = path.split(".")
        current = cfg

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = _convert_numpy_to_python(value)


class CompositeTaskGenerator(TaskGenerator):
    """Compose multiple task generators together."""

    def __init__(self, generators: List[TaskGenerator], weights: Optional[List[float]] = None):
        """
        Initialize the composite task generator.

        Args:
            generators: List of task generators to compose
            weights: Optional weights for each generator (uniform if not provided)
        """
        if not generators:
            raise ValueError("Must provide at least one generator")

        super().__init__(generators[0].base_config)

        self.generators = generators
        self.weights = weights or [1.0] * len(generators)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        logger.info(f"CompositeTaskGenerator initialized with {len(generators)} sub-generators")

    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Select and apply a generator based on weights."""
        # Select a generator
        selected_idx = rng.choice(len(self.generators), p=self.weights)
        selected_generator = self.generators[selected_idx]

        # Apply the selected generator's modifications
        selected_generator._apply_task_params(env_cfg, rng, task_id)

        # Track which generator was used
        if "task" not in env_cfg:
            env_cfg.task = {}
        env_cfg.task.generator_idx = selected_idx


def create_task_generator_from_config(config: DictConfig) -> TaskGenerator:
    """
    Factory function to create a task generator from configuration.

    Args:
        config: Configuration specifying the generator type and parameters

    Returns:
        TaskGenerator instance
    """
    generator_type = config.get("_target_", "bucketed")

    if "bucketed" in generator_type.lower():
        return BucketedTaskGenerator(
            base_config=config.get("base_config", OmegaConf.create()),
            buckets=config.get("buckets", {}),
            default_bins=config.get("default_bins", 1),
        )
    elif "random" in generator_type.lower():
        return RandomTaskGenerator(task_configs=config.get("task_configs", {}), weights=config.get("weights", None))
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
