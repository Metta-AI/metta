"""Protein optimizer adapter for sweep orchestration."""

import logging
from typing import Any

from metta.common.util.numpy_helpers import clean_numpy_types
from softmax.training.sweep.protein import Protein
from softmax.training.sweep.protein_config import ProteinConfig

logger = logging.getLogger(__name__)


class ProteinOptimizer:
    """Adapter for Protein optimizer."""

    def __init__(self, config: ProteinConfig):
        """Initialize with Protein configuration."""
        self.config = config

        # Only support Bayesian optimization
        if config.method != "bayes":
            raise ValueError(f"Unsupported optimization method: {config.method}. Only 'bayes' is supported.")

    def suggest(self, observations: list[dict[str, Any]], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions."""
        # Create fresh Protein instance (stateless)
        protein_dict = self.config.to_protein_dict()
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
            protein.observe(
                hypers=obs.get("suggestion", {}),
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
            suggestions.append(clean_numpy_types(suggestion))
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
                suggestions.append(clean_numpy_types(suggestion))
                logger.debug(f"Generated suggestion with info: {info}")

        return suggestions
