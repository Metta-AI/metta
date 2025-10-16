"""Configuration for Protein optimizer.

This module defines the Protein-specific optimization config and settings.
The canonical parameter spec (`ParameterConfig`) now lives in
`metta.sweep.core` and should be imported from there.
"""

from typing import Any, Dict, Literal

from pydantic import Field

from metta.sweep.core import ParameterConfig
from mettagrid.base_config import Config


class ProteinSettings(Config):
    """Settings for the Protein optimizer algorithm."""

    # Common settings for all methods
    max_suggestion_cost: float = Field(default=10800, description="Maximum cost for a single suggestion")
    global_search_scale: float = Field(default=1.0, description="Scale factor for global search")
    random_suggestions: int = Field(default=15, description="Number of random suggestions to generate")
    suggestions_per_pareto: int = Field(default=256, description="Number of suggestions per Pareto point")

    # Bayesian optimization specific settings
    resample_frequency: int = Field(default=10, description="How often to resample Pareto points")
    num_random_samples: int = Field(default=0, description="Number of random samples before using GP")
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
