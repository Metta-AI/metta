"""Configuration for Protein optimizer."""

from typing import Any, Dict, Literal

from pydantic import Field, model_validator

from mettagrid.base_config import Config


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


class ProteinSettings(Config):
    """Settings for the Protein optimizer algorithm."""

    # Common settings for all methods
    max_suggestion_cost: float = Field(
        default=10800, description="Maximum cost (in seconds) for a single suggestion - 3 hours for 1B timestep runs"
    )
    global_search_scale: float = Field(default=1.0, description="Scale factor for global search")
    random_suggestions: int = Field(default=10, description="Number of random suggestions to generate")
    suggestions_per_pareto: int = Field(default=256, description="Number of suggestions per Pareto point")

    # Bayesian optimization specific settings
    resample_frequency: int = Field(default=0, description="How often to resample Pareto points")
    num_random_samples: int = Field(
        default=20, description="Number of random samples before using GP - reduced for longer runs"
    )
    seed_with_search_center: bool = Field(default=True, description="Whether to seed with the search center")
    expansion_rate: float = Field(default=0.25, description="Rate of search space expansion")
    acquisition_fn: Literal["naive", "ei", "ucb"] = Field(
        default="naive", description="Acquisition function for Bayesian optimization"
    )
    ucb_beta: float = Field(default=2.0, description="Beta parameter for UCB acquisition function")
    randomize_acquisition: bool = Field(
        default=False, description="Whether to randomize acquisition function parameters for diversity"
    )


class ProteinConfig(Config):
    """Configuration for Protein hyperparameter optimization."""

    # Optimization metadata
    metric: str = Field(description="Metric to optimize (e.g., 'navigation', 'arena/combat')")
    goal: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Whether to maximize or minimize the metric"
    )
    method: Literal["bayes"] = Field(default="bayes", description="Optimization method")

    # Parameters to optimize - nested dict structure
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Nested dict of parameters to optimize")

    # Protein algorithm settings
    settings: ProteinSettings = Field(
        default_factory=ProteinSettings, description="Protein optimizer algorithm settings"
    )

    def to_protein_dict(self) -> dict:
        """Convert to the dict format expected by Protein class.

        Returns:
            Dictionary with flattened parameters and optimization settings
        """
        # Flatten the parameters into the format Protein expects
        config = {
            "metric": self.metric,
            "goal": self.goal,
            "method": self.method,
        }

        # Add parameters in the expected format
        def process_params(params: dict, prefix: str = "") -> dict:
            result = {}
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, ParameterConfig):
                    # Convert ParameterConfig to dict format
                    result[full_key] = value.model_dump()
                elif isinstance(value, dict):
                    if "min" in value and "max" in value:
                        # Already in parameter format
                        result[full_key] = value
                    else:
                        # Nested structure, recurse
                        result.update(process_params(value, full_key))
                else:
                    # Static value, not a parameter to optimize
                    pass
            return result

        # Process and add parameters
        flat_params = process_params(self.parameters)
        config.update(flat_params)

        return config
