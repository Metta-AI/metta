"""
curriculum_builder.py: The public API of curriculum creation.
"""

from itertools import product
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import config_from_path as hydra_config_from_path
from metta.mettagrid.curriculum.curriculum import (
    Curriculum,
    MettaGridTask,
)
from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithmHypers,
    DiscreteRandomHypers,
)

#
# Simplest Case: Single Task
#


def single_task(
    name: str,
    env_config: DictConfig,
) -> Curriculum:
    return task_set(
        name=name,
        env_configs=[(name, env_config)],
        env_overrides=None,
    )


#
# Task Sets (1-level Curricula)
#


def task_set(
    name: str,
    # (task_name, env_config)
    env_configs: list[tuple[str, DictConfig]],
    env_overrides: Optional[DictConfig] = None,
    curriculum_hypers: Optional[CurriculumAlgorithmHypers] = None,
    parameter_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Curriculum:
    """Creates a 1-level Curriculum with many tasks generated across a parameter grid and set of environment configs.

    Args:
        name: Name for the Curriculum root node
        env_configs: List of (task_name, env_config) tuples
        env_overrides: Optional config overrides to apply to all tasks
        curriculum_hypers: hypers for the curriculum algorithm
        parameter_ranges: Optional parameter ranges to expand via Cartesian product
            - {"values": [v1, v2, ...]} for discrete values
            - {"range": [min, max], "bins": n} for continuous ranges (bins is required)

    Returns:
        Curriculum initialized with the given configuration
    """
    if len(env_configs) == 0:
        raise ValueError("env_configs must have at least one element")

    def config_with_overrides(config: DictConfig, env_overrides: Optional[DictConfig]) -> DictConfig:
        """Create a new config with overrides applied."""
        if env_overrides is None:
            return config
        # Create a copy and merge overrides
        merged = OmegaConf.merge(config, env_overrides)
        return merged

    # If no parameter ranges specified, create tasks directly
    if parameter_ranges is None or len(parameter_ranges) == 0:
        # Create tasks with provided names
        tasks = [
            MettaGridTask(task_name, config_with_overrides(config, env_overrides)) for task_name, config in env_configs
        ]
    else:
        # Expand parameter ranges and create Cartesian product
        expanded_ranges = _expand_buckets(parameter_ranges)
        parameter_names = list(expanded_ranges.keys())
        parameter_values = list(expanded_ranges.values())

        tasks = []
        for base_name, base_config in env_configs:
            # Extract short name for the base config
            short_base_name = _get_short_name(base_name)

            for value_combination in product(*parameter_values):
                # Create config for this combination
                task_cfg = OmegaConf.create(base_config)

                # Apply parameter values
                for param, value in zip(parameter_names, value_combination, strict=False):
                    concrete_value = _sample_from_bucket_value(value)
                    OmegaConf.update(task_cfg, param, concrete_value, merge=False)

                # Apply overrides after parameter values
                task_cfg = config_with_overrides(task_cfg, env_overrides)

                # Create task name combining base name and parameter values
                param_suffix = _get_bucket_id(parameter_names, value_combination)
                if len(env_configs) == 1:
                    # Single base config: just use parameter suffix
                    task_name = param_suffix
                else:
                    # Multiple base configs: prepend short base name
                    task_name = f"{short_base_name}/{param_suffix}"

                tasks.append(MettaGridTask(task_name, task_cfg))

    if curriculum_hypers is None:
        curriculum_hypers = DiscreteRandomHypers()
    curriculum_algorithm = curriculum_hypers.create(len(tasks))

    return Curriculum(name, curriculum_algorithm, tasks)


def parameter_grid_task_set(
    name: str,
    env_cfg_template: Union[str, DictConfig],
    buckets: Dict[str, Dict[str, Any]],
    env_overrides: Optional[DictConfig] = None,
    curriculum_hypers: Optional[CurriculumAlgorithmHypers] = None,
) -> Curriculum:
    # Load base config if given as path
    if isinstance(env_cfg_template, str):
        if name == "":
            name = _get_short_name(env_cfg_template)
        base_cfg = hydra_config_from_path(env_cfg_template, None)
        assert isinstance(base_cfg, DictConfig), "Base config must be a DictConfig"
    else:
        base_cfg = env_cfg_template

    return task_set(
        name=name,
        env_configs=[(name, base_cfg)],
        env_overrides=env_overrides,
        curriculum_hypers=curriculum_hypers,
        parameter_ranges=buckets,
    )


#
# Helper functions for task set generation
#


def _expand_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Expand bucket specifications into lists of values.

    Args:
        buckets: Dict mapping parameter paths to bucket specifications

    Returns:
        Dict mapping parameter paths to lists of values
    """

    buckets_unpacked = {}
    for parameter, bucket_spec in buckets.items():
        if "values" in bucket_spec:
            buckets_unpacked[parameter] = bucket_spec["values"]
        elif "range" in bucket_spec:
            lo, hi = bucket_spec["range"]
            want_int = isinstance(lo, int) and isinstance(hi, int)

            # Check for None since ParameterRange model sets bins=None when not specified
            bins = bucket_spec.get("bins")
            if bins is None:
                # No bins specified: treat as continuous range
                buckets_unpacked[parameter] = [{"range": (lo, hi), "want_int": want_int}]
            else:
                # Bins specified: discretize the range
                n = int(bins)
                if n < 2:
                    raise ValueError(
                        f"'bins' must be >= 2 for parameter '{parameter}'. "
                        f"For continuous ranges, omit the 'bins' field."
                    )

                # Divide range into n bins
                step = (hi - lo) / n
                binned_ranges = []
                for i in range(n):
                    lo_i, hi_i = lo + i * step, lo + (i + 1) * step
                    binned_ranges.append({"range": (lo_i, hi_i), "want_int": want_int})
                buckets_unpacked[parameter] = binned_ranges
        else:
            raise ValueError(f"Invalid bucket spec: {bucket_spec}")
    return buckets_unpacked


def _sample_from_bucket_value(value: Any) -> Any:
    """Sample a concrete value from a bucket value specification."""
    import numpy as np

    if isinstance(value, dict) and "range" in value:
        lo, hi = value["range"]
        sampled = np.random.uniform(lo, hi)
        if value.get("want_int", False):
            sampled = int(sampled)
        return sampled
    return value


def _get_bucket_id(parameters: list[str], values: list[Any]) -> str:
    """Generate a unique ID for a parameter combination."""
    id_parts = []
    for param, value in zip(parameters, values, strict=False):
        # Use full parameter path to ensure uniqueness

        # Format value for ID
        if isinstance(value, dict) and "range" in value:
            lo, hi = value["range"]
            if isinstance(lo, float) or isinstance(hi, float):
                value_str = f"({lo:.3f},{hi:.3f})"
            else:
                value_str = f"({lo},{hi})"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        id_parts.append(f"{param}={value_str}")

    return ";".join(id_parts)


def _get_short_name(full_name: str) -> str:
    """Extract the short name from a full path.

    For example:
    - "/env/easy" -> "easy"
    - "path/to/navigation.yaml" -> "navigation"
    """
    # Get the last component
    name = full_name.split("/")[-1] if "/" in full_name else full_name

    # Remove file extension if present
    if "." in name:
        name = name.split(".")[0]

    return name


def curriculum_config_from_python(name_or_path: str) -> "CurriculumConfig":
    """Load a CurriculumConfig from a Python module.
    
    Args:
        name_or_path: Either:
            - A simple name like "arena_random" (looks in experiments.curriculum_defs)
            - A full module path like "experiments.curriculum_defs.arena_random"
        
    Returns:
        A CurriculumConfig instance
    """
    from metta.mettagrid.curriculum.curriculum_config import CurriculumConfig
    
    # If it's a simple name (no dots), look in the default location
    if "." not in name_or_path:
        module_path = f"curriculum_defs.{name_or_path}"
    else:
        module_path = name_or_path
    
    parts = module_path.split(".")
    module_name = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
    attr_name = parts[-1] if len(parts) > 1 else None
    
    # Ensure experiments directory is in path
    import sys
    from pathlib import Path
    experiments_path = Path(__file__).parent.parent.parent.parent.parent.parent / "experiments"
    if str(experiments_path) not in sys.path:
        sys.path.insert(0, str(experiments_path))
    
    import importlib
    try:
        module = importlib.import_module(module_name)
        if attr_name:
            config = getattr(module, attr_name)
        else:
            # If no attribute specified, module itself should be the config
            config = module
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load curriculum '{name_or_path}': {e}")
    
    if not isinstance(config, CurriculumConfig):
        raise ValueError(f"Expected CurriculumConfig, got {type(config)} from {module_path}")
    
    return config


def curriculum_config_from_path(config_path: str):
    """Load a CurriculumConfig from a config path.

    Args:
        config_path: Path to the curriculum config file

    Returns:
        A CurriculumConfig instance
    """
    from metta.mettagrid.curriculum.curriculum_config import CurriculumConfig

    cfg = hydra_config_from_path(config_path, None)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create and return CurriculumConfig
    return CurriculumConfig.model_validate(config_dict)


def curriculum_from_config_path(config_path: str, env_overrides: DictConfig) -> Curriculum:
    """Load a curriculum (Curriculum) from a config path.

    Args:
        config_path: Path to the curriculum config file
        env_overrides: Environment configuration overrides to apply

    Returns:
        A Curriculum instance representing the curriculum
    """
    # Load the CurriculumConfig
    curriculum_config = curriculum_config_from_path(config_path)

    # Create the Curriculum with env_overrides
    return curriculum_config.create(env_overrides)
