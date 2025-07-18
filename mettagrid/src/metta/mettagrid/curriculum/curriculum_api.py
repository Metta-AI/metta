"""
api.py: The public API of curriculum creation.
"""

from itertools import product
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithmHypers,
    DiscreteRandomHypers,
)
from metta.mettagrid.curriculum.task_tree import (
    MettaGridTask,
    TaskTree,
    _expand_buckets,
    _get_bucket_id,
    _get_short_name,
    _sample_from_bucket_value,
)
from metta.mettagrid.util.hydra import config_from_path

#
# Entry points to curriculum creation
#

#
# Simplest Case: Single Task
#


def single_task_tree(
    name: str,
    env_config: DictConfig,
) -> TaskTree:
    """Create a TaskTree with a single task.

    This is the simplest case - create a single task from a config,
    with no curriculum hyperparameters since there's only one task
    and no curriculum decisions to make.

    This function matches the API of the old SingleTaskCurriculum class.

    Args:
        name: Name for both the TaskTree root and the single task
        env_config: Environment configuration

    Returns:
        TaskTree with a single MettaGridTask

    Example:
        # Create a single task
        config = OmegaConf.create({"game": {"num_agents": 4}})
        tree = single_task_tree("navigation", config)
    """
    return task_set(
        name=name,
        env_configs=[(name, env_config)],
        env_overrides=None,
    )


#
# Task Sets (1-level TaskTrees)
#


def task_set(
    name: str,
    # (task_name, env_config)
    env_configs: list[tuple[str, DictConfig]],
    env_overrides: Optional[DictConfig] = None,
    curriculum_hypers: Optional[CurriculumAlgorithmHypers] = None,
    parameter_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TaskTree:
    """Helper function to create TaskTree from lists of env configs.

    Args:
        name: Name for the TaskTree root node
        env_configs: List of (task_name, env_config) tuples
        env_overrides: Optional config overrides to apply to all tasks
        curriculum_hypers: hypers for the curriculum algorithm
        parameter_ranges: Optional parameter ranges to expand via Cartesian product
            - {"values": [v1, v2, ...]} for discrete values
            - {"range": [min, max], "bins": n} for continuous ranges (bins is required)

    Returns:
        TaskTree initialized with the given configuration
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

    return TaskTree(name, curriculum_algorithm, tasks)


def parameter_grid_task_set(
    name: str,
    env_cfg_template: Union[str, DictConfig],
    buckets: Dict[str, Dict[str, Any]],
    env_overrides: Optional[DictConfig] = None,
    curriculum_hypers: Optional[CurriculumAlgorithmHypers] = None,
) -> TaskTree:
    """Create a TaskTree from bucketed parameter specifications.

    This is a convenience wrapper around task_set for the common case of
    a single base config with parameter ranges.

    Args:
        name: Name for the TaskTree root node
        env_cfg_template: Base environment config or path to config
        buckets: Dict mapping parameter paths to bucket specifications
            - {"values": [v1, v2, ...]} for discrete values
            - {"range": [min, max], "bins": n} for continuous ranges (bins >= 2)
        env_overrides: Optional config overrides to apply to all tasks
        curriculum_hypers: Hyperparameters for the curriculum algorithm

    Returns:
        TaskTree with one MettaGridTask per parameter combination
    """
    # Load base config if given as path
    if isinstance(env_cfg_template, str):
        base_cfg = config_from_path(env_cfg_template, None)
    else:
        base_cfg = env_cfg_template

    # Use task_set with parameter_ranges
    return task_set(
        name=name,
        env_configs=[("base", base_cfg)],  # Single base config
        env_overrides=env_overrides,
        curriculum_hypers=curriculum_hypers,
        parameter_ranges=buckets,
    )
