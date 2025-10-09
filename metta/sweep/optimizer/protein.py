"""Protein optimizer adapter for sweep orchestration.

This adapter remains the compatibility layer between the universal sweep
parameter spec and the Protein optimizer. It also implements categorical
parameter handling by mapping categories to integer indices before suggestions
are generated, and mapping them back after suggestions are returned.
"""

import logging
from typing import Any, Dict, Tuple

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.sweep.core import CategoricalParameterConfig, ParameterConfig
from metta.sweep.protein import Protein
from metta.sweep.protein_config import ProteinConfig

logger = logging.getLogger(__name__)


class ProteinOptimizer:
    """Adapter for Protein optimizer."""

    def __init__(self, config: ProteinConfig):
        """Initialize with Protein configuration."""
        self.config = config
        # Categorical mapping: flat_key -> (value_to_index, index_to_value)
        self._categorical_maps: dict[str, Tuple[Dict[Any, int], Dict[int, Any]]] = {}
        # Numeric-only Protein dict derived from config (categoricals converted)
        self._protein_numeric_dict: dict | None = None

        # Only support Bayesian optimization
        if config.method != "bayes":
            raise ValueError(f"Unsupported optimization method: {config.method}. Only 'bayes' is supported.")

    def suggest(self, observations: list[dict[str, Any]], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions."""
        # Lazily build the numeric-only Protein config + categorical maps
        if self._protein_numeric_dict is None:
            self._protein_numeric_dict = self._build_numeric_protein_dict_and_maps()

        protein_dict = self._protein_numeric_dict
        protein_settings = self.config.settings.model_dump()

        bayes_settings = {
            "max_suggestion_cost": protein_settings.get("max_suggestion_cost", 3600),
            "resample_frequency": protein_settings.get("resample_frequency", 0),
            "num_random_samples": protein_settings.get("num_random_samples", 50),
            "global_search_scale": protein_settings.get("global_search_scale", 1.0),
            "random_suggestions": protein_settings.get("random_suggestions", 1024),
            "suggestions_per_pareto": protein_settings.get("suggestions_per_pareto", 256),
            "seed_with_search_center": protein_settings.get("seed_with_search_center", True),
            "expansion_rate": protein_settings.get("expansion_rate", 0.25),
            "acquisition_fn": protein_settings.get("acquisition_fn", "naive"),
            "ucb_beta": protein_settings.get("ucb_beta", 2.0),
            "randomize_acquisition": protein_settings.get("randomize_acquisition", False),
        }

        protein = Protein(protein_dict, **bayes_settings)

        # Load all observations
        for obs in observations:
            sugg = obs.get("suggestion", {})
            encoded = self._encode_categoricals(sugg)
            protein.observe(
                hypers=encoded,
                score=obs.get("score", 0.0),
                cost=obs.get("cost", 0.0),
                is_failure=False,  # We don't track failures currently
            )

        logger.info(f"Loaded {len(observations)} observations into Protein optimizer")

        # Handle edge case of requesting zero suggestions
        if n_suggestions == 0:
            logger.debug("Zero suggestions requested, returning empty list")
            return []

        # Generate requested number of suggestions
        result = protein.suggest(n_suggestions=n_suggestions)

        # Handle return format: Protein returns (suggestion, info) for n=1 or [(suggestion, info), ...] for n>1
        suggestions = []

        if n_suggestions == 1:
            # Single suggestion case: returns (suggestion, info)
            suggestion, info = result
            decoded = self._decode_categoricals(suggestion)
            suggestions.append(clean_numpy_types(decoded))
            logger.debug(f"Generated suggestion with info: {info}")
        else:
            # Multiple suggestions case
            # GP-based path - got list of (suggestion, info) tuples
            result_len = len(result) if hasattr(result, "__len__") else "N/A"
            logger.debug(f"Protein.suggest returned result type: {type(result)}, length: {result_len}")
            if result:
                logger.debug(f"First element type: {type(result[0])}, value: {result[0]}")

            for item in result:
                if not isinstance(item, tuple) or len(item) != 2:
                    logger.error(f"Unexpected result format from Protein.suggest: {item}")
                    raise ValueError(f"Expected (suggestion, info) tuple, got: {item}")
                suggestion, info = item
                decoded = self._decode_categoricals(suggestion)
                suggestions.append(clean_numpy_types(decoded))
                logger.debug(f"Generated suggestion with info: {info}")

        return suggestions

    # --- Internal helpers ---
    def _build_numeric_protein_dict_and_maps(self) -> dict:
        """Create a Protein dict with categorical params converted to int_uniform.

        Returns:
            A flattened dict in the format expected by Protein
        """

        # Walk nested parameters; build new nested dict with conversions applied
        def convert_params(params: dict, prefix: str = "") -> dict:
            out: dict = {}
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, ParameterConfig):
                    out[key] = value
                elif isinstance(value, CategoricalParameterConfig):
                    choices = list(value.choices)
                    if len(choices) == 0:
                        raise ValueError(f"Categorical parameter '{full_key}' must have at least one choice")
                    value_to_index = {v: i for i, v in enumerate(choices)}
                    index_to_value = {i: v for i, v in enumerate(choices)}
                    self._categorical_maps[full_key] = (value_to_index, index_to_value)

                    # Build an equivalent numeric parameter (int_uniform)
                    n = max(1, len(choices))
                    max_idx = max(0, n - 1)
                    mean_idx = (max_idx) / 2.0
                    out[key] = ParameterConfig(
                        min=0,
                        max=max_idx,
                        distribution="int_uniform",
                        mean=mean_idx,
                        scale="auto",
                    )
                elif isinstance(value, dict):
                    out[key] = convert_params(value, full_key)
                else:
                    # Static values or unsupported types: pass through
                    pass
            return out

        numeric_params = convert_params(self.config.parameters)
        # Build a temporary ProteinConfig with numeric-only parameters
        numeric_config = ProteinConfig(
            metric=self.config.metric,
            goal=self.config.goal,
            parameters=numeric_params,
            settings=self.config.settings,
        )
        protein_dict = numeric_config.to_protein_dict()
        return protein_dict

    def _encode_categoricals(self, suggestion: dict) -> dict:
        """Map categorical values to indices using learned maps."""

        def recurse(obj: Any, prefix: str = "") -> Any:
            if not isinstance(obj, dict):
                return obj
            out: dict = {}
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out[k] = recurse(v, full_key)
                else:
                    if full_key in self._categorical_maps:
                        value_to_index, _ = self._categorical_maps[full_key]
                        out[k] = int(value_to_index.get(v, 0))
                    else:
                        out[k] = v
            return out

        return recurse(suggestion)

    def _decode_categoricals(self, suggestion: dict) -> dict:
        """Map numeric indices back to categorical values."""

        def recurse(obj: Any, prefix: str = "") -> Any:
            if not isinstance(obj, dict):
                return obj
            out: dict = {}
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out[k] = recurse(v, full_key)
                else:
                    if full_key in self._categorical_maps:
                        _, index_to_value = self._categorical_maps[full_key]
                        try:
                            idx = int(round(float(v)))
                        except Exception:
                            idx = 0
                        out[k] = index_to_value.get(idx, index_to_value.get(0))
                    else:
                        out[k] = v
            return out

        return recurse(suggestion)
