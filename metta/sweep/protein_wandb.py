import json
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone

import wandb

from metta.util.numpy.clean_numpy_types import clean_numpy_types

from .protein import Protein

logger = logging.getLogger("wandb_protein")

# Ensure appropriate logging level for debugging observation loading issues
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


class WandbProtein:
    def __init__(
        self,
        protein: Protein,
        wandb_run=None,
    ):
        """
        Initialize WandbProtein with a Protein instance and optionally a wandb run.

        Args:
            protein (Protein): The Protein instance to use for suggestions.
            wandb_run (wandb.Run, optional): The wandb run to use. If None, uses the current run.
        """
        self._wandb_run = wandb_run or wandb.run
        assert self._wandb_run is not None, "No active wandb run found"

        # Get sweep ID - handle both sweep_id attribute and sweep.id fallback
        self._sweep_id = getattr(self._wandb_run, "sweep_id", None)
        if self._sweep_id is None and hasattr(self._wandb_run, "sweep") and self._wandb_run.sweep:
            self._sweep_id = self._wandb_run.sweep.id

        logger.info(f"WandbProtein sweep ID: {self._sweep_id}")

        self._api = wandb.Api()

        self._protein = protein
        self._num_observations = 0
        self._num_failures = 0
        self._num_running = 0
        self._defunct = 0
        self._invalid = 0
        self._observations = []
        self._suggestion_info = {}  # Store info from protein.suggest()

        assert self._wandb_run.summary.get("protein.state") is None, (
            f"Run {self._wandb_run.name} already has protein state"
        )

        self._wandb_run.summary.update({"protein.state": "initializing"})

        try:
            self._load_runs()
        except Exception as e:
            logger.error(f"Error loading previous runs: {e}")
            logger.info("Continuing with fresh Protein (no previous observations)")

        try:
            self._generate_protein_suggestion()
        except Exception as e:
            logger.error(f"Error generating protein suggestion: {e}")
            # Fallback to search center if suggestion generation fails
            logger.info("Falling back to search center suggestion")
            self._suggestion = self._protein.hyperparameters.to_dict(self._protein.hyperparameters.search_centers)
            self._suggestion_info = {}

        # CRITICAL: Overwrite WandB agent's suggested parameters with Protein's suggestions
        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        # Unlock config to allow overwriting WandB agent's suggestions
        self._wandb_run.config.__dict__["_locked"] = {}

        # Set the actual parameter keys that training will use (overwriting WandB agent)
        self._wandb_run.config.update(wandb_config, allow_val_change=True)

        logger.info(f"Applied Protein suggestions to WandB config: {wandb_config}")

        # Update state to "running" after initialization is complete
        self._wandb_run.summary.update({"protein.state": "running"})
        logger.info(f"WandbProtein initialization complete for run {self._wandb_run.name}")

    def record_observation(self, objective, cost):
        """
        Record an observation (objective, cost) for the current suggestion.

        Args:
            objective (float): The objective value to optimize (higher is better for maximization).
            cost (float): The cost of this evaluation (e.g., time taken).
        """
        logger.info(f"Recording observation ({objective}, {cost}) for {self._wandb_run.name}")

        # Update WandB with the observation
        self._wandb_run.summary.update(
            {
                "protein.objective": objective,
                "protein.cost": cost,
                "protein.state": "success",
            }
        )

        # CRITICAL: Also record the observation with the Protein for future runs
        self._protein.observe(self._suggestion, objective, cost, False)
        logger.info("Observation recorded successfully in Protein for future learning")

    def record_failure(self, error_message: str = "Unknown error"):
        """
        Record that the current suggestion failed.

        Args:
            error_message (str): Description of the failure.
        """
        logger.info(f"Recording failure for {self._wandb_run.name}: {error_message}")

        # Update WandB with failure status
        self._wandb_run.summary.update(
            {
                "protein.state": "failure",
                "protein.error": error_message,
            }
        )

        # CRITICAL: Also record the failure with the Protein
        self._protein.observe(self._suggestion, 0.0, 0.0, True)

    def suggest(self, fill=None):
        """
        Get the current suggestion.

        Returns:
            tuple: (suggestion_dict, info_dict)
        """
        return self._transform_suggestion(deepcopy(self._suggestion)), self._suggestion_info

    def _transform_suggestion(self, suggestion):
        """Transform suggestion format. Override in subclasses if needed."""
        return suggestion

    def _load_runs(self):
        logger.info(f"Loading previous runs from sweep {self._sweep_id}")

        try:
            runs = list(self._get_runs_from_wandb())
            logger.info(f"Found {len(runs)} potential runs to process from WandB API")

            for i, run in enumerate(runs):
                logger.debug(f"Processing run {i + 1}/{len(runs)}: {run.name} (ID: {run.id})")
                self._update_protein_from_run(run)

        except Exception as e:
            logger.warning(f"Could not load previous runs: {e}")
            logger.info("Continuing with fresh Protein (no previous observations)")

        logger.info(
            "Initialized Protein with "
            + json.dumps(
                {
                    "observations": self._num_observations,
                    "failures": self._num_failures,
                    "running": self._num_running,
                    "defunct": self._defunct,
                    "invalid": self._invalid,
                }
            )
        )

    def _get_runs_from_wandb(self):
        # Skip loading runs if this is not a real sweep or in offline mode
        if not self._sweep_id:
            logger.info("No sweep ID found, skipping previous run loading")
            return []

        logger.info(f"Querying WandB API for runs in sweep {self._sweep_id}")

        # Add retry logic for network calls
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                runs = self._api.runs(
                    path=f"{self._wandb_run.entity}/{self._wandb_run.project}",
                    filters={
                        "sweep": self._sweep_id,
                        "summary_metrics.protein.state": {"$exists": True},
                        "id": {"$ne": self._wandb_run.id},
                    },
                    order="+created_at",
                )
                logger.info("Successfully fetched runs from WandB API")
                return runs

            except Exception as e:
                logger.warning(f"Network error fetching runs (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to fetch runs after {max_retries} attempts")
                    return []

    def _update_protein_from_run(self, run):
        logger.debug(f"Processing run {run.name} with state: {run.summary.get('protein.state', 'unknown')}")

        if run.summary.get("protein.state") == "initializing":
            logger.debug(f"Skipping run {run.name} because it is initializing")
            return

        if run.summary.get("protein.state") == "running":
            # Handle both formats: with and without microseconds
            heartbeat_str = run._attrs["heartbeatAt"]
            try:
                last_hb = datetime.strptime(heartbeat_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except ValueError:
                # Try without microseconds
                last_hb = datetime.strptime(heartbeat_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
                logger.debug(f"Skipping run {run.name} - no heartbeat in last 5 minutes")
                self._defunct += 1
                return
            self._num_running += 1
            return

        try:
            suggestion, info = self._suggestion_from_run(run)
            logger.debug(f"Successfully extracted suggestion from run {run.name}")
        except Exception as e:
            logger.warning(f"Failed to get suggestion from run {run.name}: {e}")
            self._invalid += 1
            return

        # Check for duplicate "running" state check
        if run.summary.get("protein.state") == "running":
            logger.debug(f"Recording suggestion run {run.name} that is still running")
            self._num_running += 1
            return

        objective = run.summary.get("protein.objective", 0)
        cost = run.summary.get("protein.cost", 0)
        is_failure = run.summary.get("protein.state") == "failure"

        # Log what we're attempting to load
        if is_failure:
            logger.info(f"Loading failed observation from run {run.name}: cost={cost}")
        else:
            logger.info(f"Loading observation from run {run.name}: objective={objective}, cost={cost}")

        logger.debug(
            f"Observation {run.name} "
            + f"{objective} / {cost} "
            + f"failure: {is_failure} "
            + str(suggestion)  # Use str() instead of json.dumps to avoid serialization issues
        )

        # CRITICAL FIX: Flatten the nested suggestion dict before passing to Protein
        # The suggestion from WandB is in nested format: {"trainer": {"optimizer": {"learning_rate": 0.0005}}}
        # But protein.observe() expects flattened format: {"trainer/optimizer/learning_rate": 0.0005}
        try:
            import pufferlib

            # Log the original suggestion for debugging
            logger.debug(f"Original suggestion from run {run.name}: {suggestion}")

            flattened_suggestion = dict(pufferlib.unroll_nested_dict(suggestion))
            logger.debug(f"Flattened suggestion for Protein: {list(flattened_suggestion.keys())}")

            # Pass flattened suggestion to Protein
            self._protein.observe(flattened_suggestion, objective, cost, is_failure)
            logger.debug(f"Successfully recorded observation in Protein for run {run.name}")

            # Only count as successful/failed AFTER recording in Protein succeeds
            if is_failure:
                self._num_failures += 1
            else:
                self._num_observations += 1

            # Only add to observations list if successfully recorded in Protein
            self._observations.append(
                {
                    "suggestion": suggestion,
                    "objective": objective,
                    "cost": cost,
                    "is_failure": is_failure,
                    "run_id": run.id,
                    "run_name": run.name,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to record observation in Protein for run {run.name}: {e}")
            logger.warning(f"Suggestion was: {suggestion}")
            logger.warning(f"Expected parameters: {list(self._protein.hyperparameters.flat_spaces.keys())}")
            # Increment invalid count since we couldn't learn from this observation
            self._invalid += 1

        self._suggestion_info = info

    def _suggestion_from_run(self, run):
        """
        Extract parameters from a run config - just retrieve what we stored.
        """
        logger.debug(f"Extracting suggestion from run {run.name}")

        # First try to get the flattened suggestion (new format)
        suggestion = run.summary.get("protein.suggestion_flattened", None)

        if suggestion:
            logger.debug(f"Found flattened suggestion for run {run.name}")
            # The flattened suggestion is already in the format Protein expects
            # But we need to return it in nested format for consistency
            # Reconstruct nested structure from flattened keys
            nested_suggestion = {}
            for key, value in suggestion.items():
                if "/" in key:
                    parts = key.split("/")
                    current = nested_suggestion
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    nested_suggestion[key] = value
            suggestion = nested_suggestion
        else:
            # Fall back to old format
            suggestion = run.summary.get("protein.suggestion", {})
            logger.debug(f"Raw suggestion from run {run.name}: {type(suggestion)} = {suggestion}")

            # Handle case where WandB stored the suggestion as a string
            if isinstance(suggestion, str):
                try:
                    # Try to parse the string back to a dictionary
                    import ast

                    suggestion = ast.literal_eval(suggestion)
                    logger.debug(f"Successfully parsed suggestion string for run {run.name}")
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse suggestion string from run {run.name}: {e}")
                    suggestion = {}

            # Fallback: try to extract from run config if summary doesn't have suggestion
            if not suggestion:
                logger.warning(f"No protein.suggestion found in summary for run {run.name}, trying config fallback")
                # Try to extract from the actual WandB config as fallback
                try:
                    config_dict = dict(run.config)
                    # Filter out WandB internal keys and get actual parameters
                    # Need to reconstruct the nested structure from the config
                    suggestion = {}
                    for k, v in config_dict.items():
                        if k.startswith("_") or k == "dummy_param":
                            continue
                        # Handle flattened keys like "trainer/optimizer/learning_rate"
                        if "/" in k:
                            parts = k.split("/")
                            current = suggestion
                            for part in parts[:-1]:
                                if part not in current:
                                    current[part] = {}
                                current = current[part]
                            current[parts[-1]] = v
                        else:
                            suggestion[k] = v
                    logger.debug(f"Extracted suggestion from config for run {run.name}: {suggestion}")
                except Exception as e:
                    logger.warning(f"Could not extract suggestion from config for run {run.name}: {e}")
                    suggestion = {}

        # Get the stored prediction info
        info = run.summary.get("protein.suggestion_info", {})
        logger.debug(f"Raw info from run {run.name}: {type(info)} = {info}")

        # Handle case where info might also be stored as a string
        if isinstance(info, str):
            try:
                import ast

                info = ast.literal_eval(info)
                logger.debug(f"Successfully parsed info string for run {run.name}")
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse info string from run {run.name}")
                info = {}

        # Ensure suggestion_uuid is set in info
        if "suggestion_uuid" not in info:
            info["suggestion_uuid"] = run.id

        logger.debug(f"Final extracted suggestion from run {run.name}: {suggestion}")
        return suggestion, info

    def _generate_protein_suggestion(self):
        """Generate a suggestion from Protein and store it."""
        # Pass None as fill parameter if we don't have suggestion_info yet
        fill = self._suggestion_info if hasattr(self, "_suggestion_info") and self._suggestion_info else None
        suggestion, info = self._protein.suggest(fill)
        self._suggestion = suggestion
        self._suggestion_info = info

        # Clean the suggestion and info for JSON serialization before saving to WandB
        # The _transform_suggestion method should handle cleaning for the suggestion
        cleaned_suggestion = self._transform_suggestion(deepcopy(suggestion))

        # IMPORTANT: Also save the flattened version for consistent loading
        # This ensures future runs can load observations without format issues
        import pufferlib

        flattened_suggestion = dict(pufferlib.unroll_nested_dict(suggestion))

        # Clean the flattened suggestion using clean_numpy_types directly to avoid recursion
        cleaned_flattened = clean_numpy_types(flattened_suggestion)

        # For info, use clean_numpy_types directly as well
        cleaned_info = clean_numpy_types(info)

        # Save cleaned versions to wandb summary for debugging and sweep tracking
        self._wandb_run.summary.update(
            {
                "protein.suggestion": cleaned_suggestion,
                "protein.suggestion_flattened": cleaned_flattened,  # Save cleaned flattened version
                "protein.suggestion_info": cleaned_info,
            }
        )


def create_sweep(sweep_name: str, wandb_entity: str, wandb_project: str):
    """
    Create a new wandb sweep with a dummy parameter (Protein will control all suggestions).

    Args:
        sweep_name (str): The name of the sweep.
        wandb_entity (str): The wandb entity (username or team name).
        wandb_project (str): The wandb project name.

    Returns:
        str: The ID of the created sweep.
    """
    sweep_id = wandb.sweep(
        sweep={
            "name": sweep_name,
            "method": "bayes",  # This won't actually be used since we override suggestions
            "metric": {"name": "protein.objective", "goal": "maximize"},
            "parameters": {
                # WandB requires at least one parameter, but Protein will override all suggestions
                "dummy_param": {"values": [1]}
            },
        },
        project=wandb_project,
        entity=wandb_entity,
    )
    return sweep_id
