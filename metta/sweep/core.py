"""Simplified sweep configuration API.

This module hosts the canonical parameter configuration types used to define
hyperparameter search spaces, along with convenience builders and a thin
factory (`make_sweep`) for constructing sweep tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from metta.sweep.parameter_config import CategoricalParameterConfig, Distribution, ParameterConfig, ParameterSpec
from metta.sweep.protein_config import ProteinSettings

if TYPE_CHECKING:
    from metta.tools.sweep import SweepTool


class SweepParameters:
    """Parameter configuration builder and standard presets."""

    @staticmethod
    def param(
        name: str,
        distribution: Distribution,
        min: float,
        max: float,
        search_center: float | None = None,
        scale: str = "auto",
    ) -> Dict[str, ParameterConfig]:
        """Create a custom parameter; ParameterConfig handles validation/defaults."""
        kwargs: dict[str, object] = {
            "min": min,
            "max": max,
            "distribution": distribution.value,
            "scale": scale,
        }
        if search_center is not None:
            kwargs["mean"] = search_center
        return {name: ParameterConfig(**kwargs)}

    @staticmethod
    def categorical(
        name: str,
        choices: List[Any],
    ) -> Dict[str, CategoricalParameterConfig]:
        """Create a categorical parameter.

        Args:
            name: Parameter name (e.g., "model.color").
            choices: Ordered list of allowed categorical values.

        Returns:
            Dict with single key-value pair: {name: CategoricalParameterConfig}
        """
        if not choices:
            raise ValueError("Categorical choices must be a non-empty list")
        if len(set(choices)) != len(choices):
            raise ValueError("Categorical choices must be unique")
        return {name: CategoricalParameterConfig(choices=choices)}

    # Learning rate
    LEARNING_RATE = {
        "trainer.optimizer.learning_rate": ParameterConfig(
            min=1e-5,
            max=1e-2,
            distribution="log_normal",
            mean=1e-3,
            scale="auto",
        )
    }

    # End of presets

    # PPO specific parameters
    PPO_CLIP_COEF = {
        "trainer.graph.nodes.ppo_actor.clip_coef": ParameterConfig(
            min=0.05,
            max=0.3,
            distribution="uniform",
            mean=0.2,
            scale="auto",
        )
    }

    PPO_ENT_COEF = {
        "trainer.graph.nodes.ppo_actor.ent_coef": ParameterConfig(
            min=0.0001,
            max=0.03,
            distribution="log_normal",
            mean=0.01,
            scale="auto",
        )
    }

    PPO_GAE_LAMBDA = {
        "trainer.advantage.gae_lambda": ParameterConfig(
            min=0.8,
            max=0.99,
            distribution="uniform",
            mean=0.95,
            scale="auto",
        )
    }

    PPO_VF_COEF = {
        "trainer.graph.nodes.ppo_critic.vf_coef": ParameterConfig(
            min=0.1,
            max=1.0,
            distribution="uniform",
            mean=0.5,
            scale="auto",
        )
    }

    # Optimizer parameters
    ADAM_EPS = {
        "trainer.optimizer.eps": ParameterConfig(
            min=1e-8,
            max=1e-4,
            distribution="log_normal",
            mean=1e-6,
            scale="auto",
        )
    }


def make_sweep(
    name: str,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    metric_key: str,
    search_space: Union[Dict[str, ParameterSpec], List[Dict[str, ParameterSpec]]],
    cost_key: Optional[str] = None,
    max_trials: int = 10,
    num_parallel_trials: int = 1,
    eval_overrides: Optional[Dict] = None,
    goal: Literal["maximize", "minimize"] = "maximize",
    max_concurrent_evals: Optional[int] = None,
    liar_strategy: Literal["best", "mean", "worst"] = "best",
) -> SweepTool:
    """Create a sweep with minimal configuration.

    Args (all passed as tool overrides downstream):
        Tool overrides:
            name: Sweep identifier
            recipe: Recipe module path
            train_entrypoint: Training entrypoint function
            eval_entrypoint: Evaluation entrypoint function
            metric_key: Metric to optimize
            cost_key: Optional metric path to extract cost from run summary.
                If provided, the cost will be read from summary[cost_key].
                If not provided, defaults to run.cost (which is 0 if not set).
            num_trials: Number of trials
            num_parallel_trials: Max parallel jobs
            eval_overrides: Optional overrides for evaluation configuration
            goal: Whether to maximize or minimize the metric.
            max_concurrent_evals: Maximum simultaneous evals (defaults to min(2, num_parallel_trials)).
            liar_strategy: Liar strategy for async capped scheduler.

        Protein config args:
            metric_key: Metric to optimize
            search_space: Parameters to sweep - either dict or list of single-item dicts

    Returns:
        Configured SweepTool
    """
    # Convert list of single-item dicts to flat dict
    if isinstance(search_space, list):
        flat_params = {}
        for item in search_space:
            if not isinstance(item, dict):
                raise ValueError(f"List items must be dicts, got {type(item)}")
            if len(item) != 1:
                raise ValueError(f"Each dict in list must have exactly one key-value pair, got {len(item)} keys")
            flat_params.update(item)
        search_space = flat_params

    # Keep local imports: SweepSchedulerType, SweepTool are slow loading
    from metta.tools.sweep import SweepSchedulerType, SweepTool

    protein_goal = goal
    protein_settings = ProteinSettings()

    scheduler_type = SweepSchedulerType.ASYNC_CAPPED
    scheduler_config = {
        "max_concurrent_evals": (
            max_concurrent_evals if max_concurrent_evals is not None else min(2, num_parallel_trials)
        ),
        "liar_strategy": liar_strategy,
    }

    return SweepTool(
        sweep_name=name,
        protein_metric=metric_key,
        protein_goal=protein_goal,
        protein_settings=protein_settings,
        search_space=search_space,
        recipe_module=recipe,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        max_trials=max_trials,
        max_parallel_jobs=num_parallel_trials,
        scheduler_type=scheduler_type,
        eval_overrides=eval_overrides or {},
        cost_key=cost_key,
        **scheduler_config,
    )


def grid_search(
    name: str,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    metric_key: str,
    search_space: Union[Dict[str, Any], List[Dict[str, Any]]],
    max_trials: int = 10,
    num_parallel_trials: int = 1,
    eval_overrides: Optional[Dict] = None,
    # Catch all for un-exposed tool overrides.
    # See SweepTool definition for details.
    **advanced,
) -> SweepTool:
    """Create a grid-search sweep with minimal configuration.

    Mirrors `make_sweep` but selects the grid-search scheduler and accepts
    only categorical parameters (CategoricalParameterConfig or lists). Numeric
    ParameterConfig entries are not required and are ignored if present.

    Args (all passed as tool overrides downstream):
        Tool overrides:
            name: Sweep identifier
            recipe: Recipe module path
            train_entrypoint: Training entrypoint function
            eval_entrypoint: Evaluation entrypoint function
            metric_key: Metric to optimize (used by evaluation hooks)
            max_trials: Maximum number of trials to schedule (cap on grid size)
            num_parallel_trials: Max parallel jobs
            eval_overrides: Optional overrides for evaluation configuration
            **advanced: Additional SweepTool options

        Grid parameters:
            search_space: Nested dict of categorical choices (lists or CategoricalParameterConfig)

    Returns:
        Configured SweepTool
    """
    # Convert list of single-item dicts to flat dict
    if isinstance(search_space, list):
        flat_params: Dict[str, Any] = {}
        for item in search_space:
            if not isinstance(item, dict):
                raise ValueError(f"List items must be dicts, got {type(item)}")
            if len(item) != 1:
                raise ValueError(f"Each dict in list must have exactly one key-value pair, got {len(item)} keys")
            flat_params.update(item)
        search_space = flat_params

    # Keep local imports: SweepSchedulerType, SweepTool are slow loading
    from metta.tools.sweep import SweepSchedulerType, SweepTool

    scheduler_type = SweepSchedulerType.GRID_SEARCH

    # No additional scheduler-config knobs for grid search beyond tool kwargs
    scheduler_config: Dict[str, Any] = {}

    return SweepTool(
        sweep_name=name,
        # Do not construct a ProteinConfig; grid path uses grid_parameters + grid_metric
        recipe_module=recipe,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        max_trials=max_trials,
        max_parallel_jobs=num_parallel_trials,
        scheduler_type=scheduler_type,
        eval_overrides=eval_overrides or {},
        grid_parameters=search_space,  # categorical choices
        grid_metric=metric_key,
        **scheduler_config,
        **advanced,
    )
