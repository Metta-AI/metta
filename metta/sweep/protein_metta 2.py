import logging
from typing import Any, Tuple

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.sweep.protein_config import ProteinConfig

from .protein import ParetoGenetic, Protein, Random

logger = logging.getLogger("metta_protein")

# Ensure appropriate logging level for debugging observation loading issues
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


class MettaProtein:
    """MettaProtein is a thin wrapper around Protein that allows for easy integration with the rest of the codebase.
    It is used to generate suggestions and record observations."""

    def __init__(
        self,
        cfg: ProteinConfig,
    ):
        self._cfg = cfg

        # Convert ProteinConfig to the dict format Protein expects
        protein_dict = cfg.to_protein_dict()

        # Get protein settings as dict
        protein_settings = cfg.settings.model_dump()

        # Initialize the appropriate optimizer based on method
        if cfg.method == "random":
            # Random uses a subset of settings
            random_settings = {
                "global_search_scale": protein_settings.get("global_search_scale", 1.0),
                "random_suggestions": protein_settings.get("random_suggestions", 1024),
                "acquisition_fn": protein_settings.get("acquisition_fn", "naive"),
            }
            self._protein = Random(protein_dict, **random_settings)
        elif cfg.method == "genetic":
            # ParetoGenetic uses a subset of settings
            genetic_settings = {
                "global_search_scale": protein_settings.get("global_search_scale", 1.0),
                "suggestions_per_pareto": protein_settings.get("suggestions_per_pareto", 256),
                "bias_cost": protein_settings.get("bias_cost", True),
                "log_bias": protein_settings.get("log_bias", False),
                "acquisition_fn": protein_settings.get("acquisition_fn", "naive"),
            }
            self._protein = ParetoGenetic(protein_dict, **genetic_settings)
        else:  # bayes (default)
            # Protein uses most settings but not genetic-specific ones
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
            self._protein = Protein(protein_dict, **bayes_settings)

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
        # Only Protein class has failure_observations, Random and ParetoGenetic don't
        success_count = len(getattr(self._protein, "success_observations", []))
        failure_count = len(getattr(self._protein, "failure_observations", []))
        return success_count + failure_count
