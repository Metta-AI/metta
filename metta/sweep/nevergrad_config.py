"""Configuration for Nevergrad optimizer adapter.

Defines a NevergradConfig that mirrors our canonical ParameterSpec
and contains adapter-specific settings for optimizer selection and budget.
"""

from typing import Any, Dict, Literal

from pydantic import Field

from metta.sweep.core import ParameterSpec
from mettagrid.base_config import Config


class NevergradSettings(Config):
    """Settings for Nevergrad optimizer selection and runtime."""

    optimizer_name: str = Field(default="NGOpt", description="Name of the Nevergrad optimizer to use")
    budget: int = Field(default=100, description="Total evaluation budget for the optimizer")
    num_workers: int = Field(default=1, description="Parallelism hint for the optimizer")
    seed: int | None = Field(default=None, description="Optional random seed for reproducibility")


class NevergradConfig(Config):
    """Configuration for Nevergrad hyperparameter optimization."""

    # Optimization metadata
    metric: str = Field(description="Metric to optimize")
    goal: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization direction")

    # Parameters to optimize
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Nested dict of parameters: values are ParameterConfig or CategoricalParameterConfig or nested dicts"
        ),
    )

    # Nevergrad algorithm settings
    settings: NevergradSettings = Field(default_factory=NevergradSettings)

    @classmethod
    def from_parameters(
        cls,
        *,
        metric: str,
        goal: Literal["maximize", "minimize"],
        parameters: Dict[str, ParameterSpec] | list[dict[str, ParameterSpec]],
        settings: NevergradSettings | None = None,
    ) -> "NevergradConfig":
        """Helper to build a NevergradConfig from canonical parameters.

        Accepts either a flat dict or a list of single-entry dicts (like make_sweep).
        """
        # Normalize list inputs to a single flat dict
        if isinstance(parameters, list):
            flat_params: dict[str, ParameterSpec] = {}
            for item in parameters:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError("Each element must be a single-entry dict of {name: ParameterSpec}")
                flat_params.update(item)
            parameters = flat_params
        return cls(metric=metric, goal=goal, parameters=parameters, settings=settings or NevergradSettings())
