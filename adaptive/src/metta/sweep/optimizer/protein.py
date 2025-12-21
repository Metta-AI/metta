"""Protein optimizer adapter for sweep orchestration with flat dot-path parameters."""

import logging
from typing import Any, Dict, Tuple

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.sweep.parameter_config import CategoricalParameterConfig, ParameterConfig
from metta.sweep.protein import Protein
from metta.sweep.protein_config import ProteinConfig

logger = logging.getLogger(__name__)


class ProteinOptimizer:
    """Adapter for Protein optimizer using flat dot-path parameters."""

    def __init__(self, config: ProteinConfig):
        """Initialize with Protein configuration."""
        self.config = config
        # Categorical mapping: flat_key -> (value_to_index, index_to_value)
        self._categorical_maps: dict[str, Tuple[Dict[Any, int], Dict[int, Any]]] = {}
        # Numeric-only Protein dict derived from config (categoricals converted)
        self._protein_numeric_dict: dict | None = None

        if config.method != "bayes":
            raise ValueError(f"Unsupported optimization method: {config.method}. Only 'bayes' is supported.")

    def suggest(self, observations: list[dict[str, Any]], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions."""
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

        for obs in observations:
            sugg = obs.get("suggestion", {})
            encoded = self._encode_categoricals(sugg)
            protein.observe(
                hypers=encoded,
                score=obs.get("score", 0.0),
                cost=obs.get("cost", 0.0),
                is_failure=bool(obs.get("is_failure", False)),
            )

        logger.info(f"Loaded {len(observations)} observations into Protein optimizer")

        if n_suggestions == 0:
            logger.debug("Zero suggestions requested, returning empty list")
            return []

        result = protein.suggest(n_suggestions=n_suggestions)
        suggestions: list[dict[str, Any]] = []

        if n_suggestions == 1:
            suggestion, info = result
            decoded = self._decode_categoricals(suggestion)
            suggestions.append(clean_numpy_types(decoded))
            logger.debug(f"Generated suggestion with info: {info}")
        else:
            result_len = len(result) if hasattr(result, "__len__") else "N/A"
            logger.debug(f"Protein.suggest returned result type: {type(result)}, length: {result_len}")
            if result:
                logger.debug(f"First element type: {type(result[0])}, value: {result[0]}")

            for item in result:
                if not isinstance(item, tuple) or len(item) != 2:
                    logger.error(f"Unexpected result format from Protein.suggest: {item}", exc_info=True)
                    raise ValueError(f"Expected (suggestion, info) tuple, got: {item}")
                suggestion, info = item
                decoded = self._decode_categoricals(suggestion)
                suggestions.append(clean_numpy_types(decoded))
                logger.debug(f"Generated suggestion with info: {info}")

        return suggestions

    # --- Internal helpers ---
    def _build_numeric_protein_dict_and_maps(self) -> dict:
        """Create a Protein dict with categorical params converted to int_uniform.

        Parameters are expected to be flat dot-path keys.
        """
        numeric_params: dict[str, Any] = {}
        for key, value in self.config.parameters.items():
            if isinstance(value, ParameterConfig):
                numeric_params[key] = value
            elif isinstance(value, CategoricalParameterConfig):
                choices = list(value.choices)
                if len(choices) == 0:
                    raise ValueError(f"Categorical parameter '{key}' must have at least one choice")
                value_to_index = {v: i for i, v in enumerate(choices)}
                index_to_value = {i: v for i, v in enumerate(choices)}
                self._categorical_maps[key] = (value_to_index, index_to_value)
                n = max(1, len(choices))
                max_idx = max(0, n - 1)
                mean_idx = (max_idx) / 2.0
                numeric_params[key] = ParameterConfig(
                    min=0,
                    max=max_idx,
                    distribution="int_uniform",
                    mean=mean_idx,
                    scale="auto",
                )
            else:
                # Static values or unsupported types: ignore
                pass

        numeric_config = ProteinConfig(
            metric=self.config.metric,
            goal=self.config.goal,
            parameters=numeric_params,
            settings=self.config.settings,
        )
        protein_dict = numeric_config.to_protein_dict()
        return protein_dict

    def _encode_categoricals(self, suggestion: dict) -> dict:
        """Map categorical values to indices using learned maps (flat keys)."""
        encoded: dict[str, Any] = {}
        for k, v in suggestion.items():
            if k in self._categorical_maps:
                value_to_index, _ = self._categorical_maps[k]
                if v not in value_to_index:
                    logger.warning(f"Unknown categorical value '{v}' for parameter '{k}', defaulting to first choice.")
                encoded[k] = int(value_to_index.get(v, 0))
            else:
                encoded[k] = v
        return encoded

    def _decode_categoricals(self, suggestion: dict) -> dict:
        """Map numeric indices back to categorical values (flat keys)."""
        decoded: dict[str, Any] = {}
        for k, v in suggestion.items():
            if k in self._categorical_maps:
                _, index_to_value = self._categorical_maps[k]
                try:
                    idx = int(round(float(v)))
                except Exception:
                    idx = 0
                decoded[k] = index_to_value.get(idx, index_to_value.get(0))
            else:
                decoded[k] = v
        return decoded
