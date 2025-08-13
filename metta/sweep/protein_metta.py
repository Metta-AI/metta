import logging
from typing import Any, Tuple, cast

from omegaconf import DictConfig, OmegaConf

from metta.common.util.numpy_helpers import clean_numpy_types

from .protein_advanced import ProteinAdvanced

logger = logging.getLogger("wandb_protein")

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
        cfg: DictConfig,
    ):
        self._cfg = cfg

        # Convert OmegaConf to regular dict
        parameters_dict = cast(dict[str, Any], OmegaConf.to_container(cfg.parameters, resolve=True))

        # Create the config structure that Protein expects
        protein_config = {
            "metric": cfg.metric,
            "goal": cfg.goal,
            "method": cfg.method,
            **parameters_dict,  # Add flattened parameters at top level
        }
        raw_settings = cast(dict[str, Any], OmegaConf.to_container(cfg.protein, resolve=True))

        # Filter unsupported keys and map deprecated ones
        allowed_keys = {
            "acquisition_fn",
            "max_suggestion_cost",
            "num_random_samples",
            "global_search_scale",
            "random_suggestions",
            "suggestions_per_pareto",
            "seed_with_search_center",
            "expansion_rate",
            "constraint_tolerance",
            "multi_fidelity",
            "beta_ucb",
            "xi_ei",
        }

        protein_settings: dict[str, Any] = {}
        for k, v in raw_settings.items():
            if k == "resample_frequency":
                # Deprecated in advanced optimizer; ignore
                continue
            if k == "phase":
                # Metadata only; ignore for constructor
                continue
            if k in allowed_keys:
                protein_settings[k] = v

        # Initialize Protein with sweep config and protein-specific settings
        self._protein = ProteinAdvanced(protein_config, **protein_settings)

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
        # ProteinAdvanced stores unified observations
        if hasattr(self._protein, "observations"):
            return len(self._protein.observations)  # type: ignore[attr-defined]
        # Fallback for legacy implementations
        success = getattr(self._protein, "success_observations", [])
        failure = getattr(self._protein, "failure_observations", [])
        return len(success) + len(failure)
