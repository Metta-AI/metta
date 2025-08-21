import logging
from typing import Any, Tuple

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.sweep.protein_config import ProteinConfig

from .protein import Protein

logger = logging.getLogger("metta_protein")

# Ensure appropriate logging level for debugging observation loading issues
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


class MettaProtein:
    """
    MettaProtein is a thin wrapper around Protein that allows for easy integration with the rest of the codebase.
    It is used to generate suggestions and record observations.
    """

    def __init__(
        self,
        cfg: ProteinConfig,
    ):
        self._cfg = cfg

        # Convert ProteinConfig to the dict format Protein expects
        protein_dict = cfg.to_protein_dict()

        # Get protein settings as dict
        protein_settings = cfg.settings.model_dump()

        # Initialize Protein with sweep config and protein-specific settings
        self._protein = Protein(protein_dict, **protein_settings)

    def suggest(self, fill=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Get the current suggestion as a nested dictionary.

        Returns:
            Tuple[dict[str, Any], dict[str, Any]]: A tuple containing:
                - suggestion: A nested dictionary of hyperparameters (e.g., {"trainer": {"lr": 0.001}})
                - info: Additional information about the suggestion
        """
        suggestion, info = self._protein.suggest(fill)
        return clean_numpy_types(suggestion), info

    def observe(self, suggestion: dict[str, Any], objective: float, cost: float, is_failure: bool) -> None:
        """
        Record an observation.

        Args:
            suggestion (dict[str, Any]): The suggestion to record.
            objective (float): The objective value to optimize.
            cost (float): The cost of this evaluation (e.g., time taken).
            is_failure (bool): Whether the suggestion failed.
        """
        self._protein.observe(suggestion, objective, cost, is_failure)

    def observe_failure(self, suggestion: dict[str, Any]) -> None:
        """
        Record a failure.
        """
        self._protein.observe(suggestion, 0, 0.01, True)

    @property
    def num_observations(self) -> int:
        """
        Get the number of observations.
        """
        return len(self._protein.success_observations) + len(self._protein.failure_observations)
