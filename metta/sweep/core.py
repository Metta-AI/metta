"""Simplified sweep configuration API.

This module hosts the canonical parameter configuration types used to define
hyperparameter search spaces, along with convenience builders and a thin
factory (`make_sweep`) for constructing sweep tools.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from mettagrid.base_config import Config

if TYPE_CHECKING:
    # For type checking only; avoid runtime import cycles
    from metta.tools.sweep import SweepTool


class Distribution(StrEnum):
    """Supported parameter distributions."""

    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    UNIFORM_POW2 = "uniform_pow2"
    LOG_NORMAL = "log_normal"
    LOGIT_NORMAL = "logit_normal"


class ParameterConfig(Config):
    """Configuration for a single hyperparameter to optimize.

    Performs internal validation/sanitization:
    - For "logit_normal", clamps bounds to (1e-6, 1 - 1e-6)
    - If "mean" is omitted, defaults to geometric mean for log/log2 and arithmetic mean otherwise
    - Ensures min < max
    """

    min: float = Field(description="Minimum value for the parameter")
    max: float = Field(description="Maximum value for the parameter")
    distribution: Literal["uniform", "int_uniform", "uniform_pow2", "log_normal", "logit_normal"] = Field(
        description="Distribution type for sampling"
    )
    mean: float = Field(description="Mean/center value for search")
    scale: float | str = Field(description="Scale for the parameter search")

    @model_validator(mode="before")
    @classmethod
    def _sanitize_and_default(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        v = dict(values)
        dist = v.get("distribution")

        # Clamp for logit-normal to avoid 0/1 boundary issues
        if dist == "logit_normal":
            eps = 1e-6
            try:
                v_min = float(v.get("min"))
                v_max = float(v.get("max"))
            except Exception:
                return v
            v_min = max(v_min, eps)
            v_max = min(v_max, 1 - eps)
            v["min"] = v_min
            v["max"] = v_max

        # Default mean if not provided
        if v.get("mean") is None:
            try:
                v_min = float(v.get("min"))
                v_max = float(v.get("max"))
            except Exception:
                return v
            if dist in ("log_normal", "uniform_pow2"):
                v["mean"] = (v_min * v_max) ** 0.5
            else:
                v["mean"] = (v_min + v_max) / 2.0

        # Basic bound validation
        try:
            if float(v.get("min")) >= float(v.get("max")):
                raise ValueError("min must be less than max")
        except Exception:
            return v

        return v


class CategoricalParameterConfig(Config):
    """Configuration for a categorical hyperparameter.

    Optimizer adapters may map this to their native categorical representation.
    For optimizers without native categorical support, adapters may encode
    categories via indices or one-hot schemes as appropriate.
    """

    choices: List[Any] = Field(description="List of allowed categorical values")


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
        "trainer.losses.loss_configs.ppo.clip_coef": ParameterConfig(
            min=0.05,
            max=0.3,
            distribution="uniform",
            mean=0.2,
            scale="auto",
        )
    }

    PPO_ENT_COEF = {
        "trainer.losses.loss_configs.ppo.ent_coef": ParameterConfig(
            min=0.0001,
            max=0.03,
            distribution="log_normal",
            mean=0.01,
            scale="auto",
        )
    }

    PPO_GAE_LAMBDA = {
        "trainer.losses.loss_configs.ppo.gae_lambda": ParameterConfig(
            min=0.8,
            max=0.99,
            distribution="uniform",
            mean=0.95,
            scale="auto",
        )
    }

    PPO_VF_COEF = {
        "trainer.losses.loss_configs.ppo.vf_coef": ParameterConfig(
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


# Type alias for any supported parameter specification
ParameterSpec = ParameterConfig | CategoricalParameterConfig


def make_sweep(
    name: str,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    objective: str,
    parameters: Union[Dict[str, ParameterSpec], List[Dict[str, ParameterSpec]]],
    max_trials: int = 10,
    num_parallel_trials: int = 1,
    train_overrides: Optional[Dict] = None,
    eval_overrides: Optional[Dict] = None,
    # Catch all for un-exposed tool overrides.
    # See SweepTool definition for details.
    **advanced,
) -> "SweepTool":
    """Create a sweep with minimal configuration.

    Args (all passed as tool overrides downstream):
        Tool overrides:
            name: Sweep identifier
            recipe: Recipe module path
            train_entrypoint: Training entrypoint function
            eval_entrypoint: Evaluation entrypoint function
            num_trials: Number of trials
            num_parallel_trials: Max parallel jobs
            train_overrides: Optional overrides for training configuration
            eval_overrides: Optional overrides for evaluation configuration
            **advanced: Additional SweepTool options

        Protein config args:
            objective: Metric to optimize
            parameters: Parameters to sweep - either dict or list of single-item dicts

    Returns:
        Configured SweepTool
    """
    # Convert list of single-item dicts to flat dict
    if isinstance(parameters, list):
        flat_params = {}
        for item in parameters:
            if not isinstance(item, dict):
                raise ValueError(f"List items must be dicts, got {type(item)}")
            if len(item) != 1:
                raise ValueError(f"Each dict in list must have exactly one key-value pair, got {len(item)} keys")
            flat_params.update(item)
        parameters = flat_params

    # Local imports to avoid circular dependencies
    from metta.sweep.protein_config import ProteinConfig, ProteinSettings
    from metta.tools.sweep import SweepSchedulerType, SweepTool

    protein_config = ProteinConfig(
        metric=objective,
        goal=advanced.pop("goal", "maximize"),
        parameters=parameters,
        settings=ProteinSettings(),
    )

    scheduler_type = SweepSchedulerType.ASYNC_CAPPED
    scheduler_config = {
        "max_concurrent_evals": advanced.pop("max_concurrent_evals", min(2, num_parallel_trials)),
        "liar_strategy": advanced.pop("liar_strategy", "best"),
    }

    return SweepTool(
        sweep_name=name,
        protein_config=protein_config,
        recipe_module=recipe,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        max_trials=max_trials,
        max_parallel_jobs=num_parallel_trials,
        scheduler_type=scheduler_type,
        train_overrides=train_overrides or {},
        eval_overrides=eval_overrides or {},
        **scheduler_config,
        **advanced,
    )


def grid_search(
    name: str,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    objective: str,
    parameters: Union[Dict[str, Any], List[Dict[str, Any]]],
    max_trials: int = 10,
    num_parallel_trials: int = 1,
    train_overrides: Optional[Dict] = None,
    eval_overrides: Optional[Dict] = None,
    # Catch all for un-exposed tool overrides.
    # See SweepTool definition for details.
    **advanced,
) -> "SweepTool":
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
            max_trials: Maximum number of trials to schedule (cap on grid size)
            num_parallel_trials: Max parallel jobs
            train_overrides: Optional overrides for training configuration
            eval_overrides: Optional overrides for evaluation configuration
            **advanced: Additional SweepTool options

        Grid parameters:
            objective: Metric to optimize (used by evaluation hooks)
            parameters: Nested dict of categorical choices (lists or CategoricalParameterConfig)

    Returns:
        Configured SweepTool
    """
    # Convert list of single-item dicts to flat dict
    if isinstance(parameters, list):
        flat_params: Dict[str, Any] = {}
        for item in parameters:
            if not isinstance(item, dict):
                raise ValueError(f"List items must be dicts, got {type(item)}")
            if len(item) != 1:
                raise ValueError(f"Each dict in list must have exactly one key-value pair, got {len(item)} keys")
            flat_params.update(item)
        parameters = flat_params

    # Local imports to avoid circular dependencies
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
        train_overrides=train_overrides or {},
        eval_overrides=eval_overrides or {},
        grid_parameters=parameters,  # categorical choices
        grid_metric=objective,
        **scheduler_config,
        **advanced,
    )
