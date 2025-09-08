"""Protein optimizer adapter for sweep orchestration."""

import logging
from typing import Any

from metta.sweep.models import Observation
from metta.sweep.protein_config import ProteinConfig
from metta.sweep.protein_metta import MettaProtein

logger = logging.getLogger(__name__)


class ProteinOptimizer:
    """Adapter for Protein optimizer."""

    def __init__(self, config: ProteinConfig):
        """Initialize with Protein configuration."""
        self.config = config

    def suggest(self, observations: list[Observation], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Generate hyperparameter suggestions."""
        # Create fresh MettaProtein instance (stateless)
        protein = MettaProtein(self.config)

        # Load all observations
        for obs in observations:
            protein.observe(
                suggestion=obs.suggestion,
                objective=obs.score,
                cost=obs.cost,
                is_failure=False,  # We don't track failures currently
            )

        logger.info(f"Loaded {len(observations)} observations into Protein optimizer")

        # Generate requested number of suggestions
        suggestions = []
        for _ in range(n_suggestions):
            suggestion, info = protein.suggest()
            suggestions.append(suggestion)
            logger.debug(f"Generated suggestion with info: {info}")

        return suggestions
