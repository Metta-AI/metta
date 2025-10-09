"""Simplified sweep configuration API."""

from enum import StrEnum
from typing import Dict, List, Optional, Union

import numpy as np

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.tools.sweep import SweepSchedulerType, SweepTool


class Distribution(StrEnum):
    """Supported parameter distributions."""

    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    UNIFORM_POW2 = "uniform_pow2"
    LOG_NORMAL = "log_normal"
    LOGIT_NORMAL = "logit_normal"


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
        """Create a custom parameter with sanitization.

        Args:
            name: Parameter name (e.g., "trainer.optimizer.learning_rate")
            distribution: Distribution type
            min: Minimum value
            max: Maximum value
            search_center: Center point for search (defaults to mean)
            scale: Scale parameter

        Returns:
            Dict with single key-value pair: {name: ParameterConfig}
        """
        # Sanitize logit distribution bounds
        if distribution == Distribution.LOGIT_NORMAL:
            min = np.minimum(min, 1e-6)
            max = np.maximum(max, 1 - 1e-6)

        # Validate bounds
        if min >= max:
            raise ValueError(f"min ({min}) must be less than max ({max})")

        # Default search center based on distribution
        if search_center is None:
            if distribution in [Distribution.LOG_NORMAL, Distribution.UNIFORM_POW2]:
                search_center = (min * max) ** 0.5  # Geometric mean
            else:
                search_center = (min + max) / 2  # Arithmetic mean

        return {
            name: ParameterConfig(
                min=min,
                max=max,
                distribution=distribution.value,
                mean=search_center,
                scale=scale,
            )
        }

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


def make_sweep(
    name: str,
    recipe: str,
    train_entrypoint: str,
    eval_entrypoint: str,
    objective: str,
    parameters: Union[Dict[str, ParameterConfig], List[Dict[str, ParameterConfig]]],
    num_trials: int = 10,
    num_parallel_trials: int = 1,
    train_overrides: Optional[Dict] = None,
    eval_overrides: Optional[Dict] = None,
    **advanced,
) -> SweepTool:
    """Create a sweep with minimal configuration.

    Args:
        name: Sweep identifier
        recipe: Recipe module path
        train_entrypoint: Training entrypoint function
        eval_entrypoint: Evaluation entrypoint function
        objective: Metric to optimize
        parameters: Parameters to sweep - either dict or list of single-item dicts
        num_trials: Number of trials
        num_parallel_trials: Max parallel jobs
        train_overrides: Optional overrides for training configuration
        eval_overrides: Optional overrides for evaluation configuration
        **advanced: Additional SweepTool options

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

    protein_config = ProteinConfig(
        metric=objective,
        goal=advanced.pop("goal", "maximize"),
        parameters=parameters,
        settings=ProteinSettings(
            num_random_samples=0,
            max_suggestion_cost=3600 * 6,
            resample_frequency=10,
            random_suggestions=15,
            suggestions_per_pareto=128,
        ),
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
        max_trials=num_trials,
        max_parallel_jobs=num_parallel_trials,
        scheduler_type=scheduler_type,
        train_overrides=train_overrides or {},
        eval_overrides=eval_overrides or {},
        **scheduler_config,
        **advanced,
    )
