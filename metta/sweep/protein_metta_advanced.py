"""MettaProtein wrapper using the advanced Protein implementation."""

import logging
from typing import Any, Tuple

from omegaconf import DictConfig, OmegaConf

from metta.common.util.numpy_helpers import clean_numpy_types

from .protein_advanced import ProteinAdvanced

logger = logging.getLogger("wandb_protein")

# Ensure appropriate logging level for debugging observation loading issues
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


class MettaProteinAdvanced:
    """
    MettaProteinAdvanced is a wrapper around ProteinAdvanced that provides integration with the Metta codebase.

    Key improvements over MettaProtein:
    - Uses proper acquisition functions (EI, UCB)
    - Better GP handling with dimension safety
    - Cleaner observation tracking
    - Ready for multi-objective optimization
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize MettaProteinAdvanced with configuration.

        Args:
            cfg: OmegaConf configuration containing sweep parameters
        """
        self._cfg = cfg

        # Convert OmegaConf to regular dict
        parameters_dict = OmegaConf.to_container(cfg.parameters, resolve=True)

        # Create the config structure that ProteinAdvanced expects
        protein_config = {
            "metric": cfg.metric,
            "goal": cfg.goal,
            "method": cfg.method,
            **parameters_dict,  # Add flattened parameters at top level
        }

        # Extract protein settings if they exist
        protein_settings = {}
        if "protein" in cfg:
            protein_settings = OmegaConf.to_container(cfg.protein, resolve=True)

            # Map old parameter names to new ones
            # Remove parameters that don't exist in ProteinAdvanced
            if "resample_frequency" in protein_settings:
                logger.info("Ignoring deprecated parameter 'resample_frequency'")
                del protein_settings["resample_frequency"]

            # Ensure we have sensible defaults for new parameters
            if "acquisition_fn" not in protein_settings:
                protein_settings["acquisition_fn"] = "ei"  # Default to Expected Improvement
                logger.info("Using default acquisition function: Expected Improvement (ei)")

        # Initialize ProteinAdvanced with sweep config and protein-specific settings
        self._protein = ProteinAdvanced(protein_config, **protein_settings)
        logger.info(f"Initialized ProteinAdvanced with acquisition_fn={protein_settings.get('acquisition_fn', 'ei')}")

    def suggest(self, fill=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Get the current suggestion as a nested dictionary.

        Returns:
            Tuple[dict[str, Any], dict[str, Any]]: A tuple containing:
                - suggestion: A nested dictionary of hyperparameters (e.g., {"trainer": {"lr": 0.001}})
                - info: Additional information about the suggestion (may include acquisition_value)
        """
        suggestion, info = self._protein.suggest(fill)

        # Log acquisition value if available
        if "acquisition_value" in info:
            logger.debug(f"Suggestion acquisition value: {info['acquisition_value']:.4f}")

        return clean_numpy_types(suggestion), info

    def observe(self, suggestion: dict[str, Any], objective: float, cost: float, is_failure: bool) -> None:
        """
        Record an observation.

        Args:
            suggestion: The suggestion to record
            objective: The objective value to optimize
            cost: The cost of this evaluation (e.g., time taken)
            is_failure: Whether the suggestion failed
        """
        # ProteinAdvanced has the same observe signature
        self._protein.observe(suggestion, objective, cost, is_failure)

        # Log observation
        status = "FAILED" if is_failure else "SUCCESS"
        logger.debug(f"Recorded {status} observation: objective={objective:.4f}, cost={cost:.2f}")

    def observe_failure(self, suggestion: dict[str, Any]) -> None:
        """
        Record a failure.

        Args:
            suggestion: The suggestion that failed
        """
        # Use small cost for failures
        self._protein.observe(suggestion, 0, 0.01, True)
        logger.debug("Recorded failure observation")

    @property
    def num_observations(self) -> int:
        """
        Get the number of observations.

        Returns:
            int: Total number of observations (successes + failures)
        """
        # ProteinAdvanced uses a single observations list
        return len(self._protein.observations)

    @property
    def num_successful_observations(self) -> int:
        """
        Get the number of successful observations.

        Returns:
            int: Number of successful observations
        """
        return len([obs for obs in self._protein.observations if not obs.is_failure])

    @property
    def num_failed_observations(self) -> int:
        """
        Get the number of failed observations.

        Returns:
            int: Number of failed observations
        """
        return len([obs for obs in self._protein.observations if obs.is_failure])

    @property
    def best_observation(self) -> Tuple[dict[str, Any], float]:
        """
        Get the best observation so far.

        Returns:
            Tuple[dict[str, Any], float]: The best suggestion and its objective value
        """
        valid_obs = [obs for obs in self._protein.observations if not obs.is_failure]
        if not valid_obs:
            return None, float("-inf")

        # Find best based on optimization direction
        if self._protein.hyperparameters.optimize_direction == 1:  # Maximize
            best_obs = max(valid_obs, key=lambda obs: obs.objectives[0])
        else:  # Minimize
            best_obs = min(valid_obs, key=lambda obs: obs.objectives[0])

        # Convert back to suggestion dict
        suggestion = self._protein.hyperparameters.to_dict(best_obs.input)
        return suggestion, best_obs.objectives[0]


# For backward compatibility, also export as MettaProtein
MettaProtein = MettaProteinAdvanced
