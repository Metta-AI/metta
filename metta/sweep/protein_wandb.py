import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Tuple

import pufferlib
import wandb

from metta.common.util.numpy_helpers import clean_numpy_types

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
        max_runs_to_load: int = 100,
    ):
        """
        Initialize WandbProtein with a Protein instance and optionally a wandb run.

        Args:
            protein (Protein): The Protein instance to use for suggestions.
            wandb_run (wandb.Run, optional): The wandb run to use. If None, uses the current run.
            max_runs_to_load (int, optional): Maximum number of previous runs to load. Defaults to 100.
        """
        self._wandb_run = wandb_run or wandb.run
        assert self._wandb_run is not None, "No active wandb run found"

        # Get sweep ID - handle both sweep_id attribute and sweep.id fallback
        self._sweep_id = getattr(self._wandb_run, "sweep_id", None)
        if self._sweep_id is None and hasattr(self._wandb_run, "sweep") and self._wandb_run.sweep:
            self._sweep_id = self._wandb_run.sweep.id
        logger.info(f"Sweep ID: {self._sweep_id}")
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

        self._max_runs_to_load = max_runs_to_load

        try:
            self._load_runs()
        except Exception as e:
            logger.error(f"Error loading previous runs: {e}")

        # Generate protein suggestion - let exceptions propagate
        self._generate_protein_suggestion()

        # CRITICAL: Overwrite WandB agent's suggested parameters with Protein's suggestions
        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        # Unlock config to allow overwriting WandB agent's suggestions
        self._wandb_run.config.__dict__["_locked"] = {}

        # Set the actual parameter keys that training will use (overwriting WandB agent)
        self._wandb_run.config.update(wandb_config, allow_val_change=True)

        # Update state to "running" after initialization is complete
        self._wandb_run.summary.update({"protein.state": "running"})

    def record_observation(self, objective, cost):
        """
        Record an observation (objective, cost) for the current suggestion.

        Args:
            objective (float): The objective value to optimize (higher is better for maximization).
            cost (float): The cost of this evaluation (e.g., time taken).
        """
        # Update WandB with the observation
        self._wandb_run.summary.update(
            {
                "protein.objective": objective,
                "protein.cost": cost,
                "protein.state": "success",
            }
        )

        # CRITICAL: Also record the observation with the Protein for future runs
        logger.info(f"Recording observation: {self._suggestion}, {objective}, {cost}")
        self._protein.observe(self._suggestion, objective, cost, False)

    def record_failure(self, error_message: str = "Unknown error"):
        """
        Record that the current suggestion failed.

        Args:
            error_message (str): Description of the failure.
        """
        # Update WandB with failure status
        self._wandb_run.summary.update(
            {
                "protein.state": "failure",
                "protein.error": error_message,
            }
        )

        # CRITICAL: Also record the failure with the Protein
        self._protein.observe(self._suggestion, 0.0, 0.0, True)

    def suggest(self, fill=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Get the current suggestion.

        """
        # Return the already-cleaned suggestion and info
        return deepcopy(self._suggestion), deepcopy(self._suggestion_info)

    def _transform_suggestion(self, suggestion) -> dict[str, Any]:
        """Transform suggestion format. Override in subclasses if needed."""
        # Clean numpy types and any other non-serializable objects
        try:
            # First try with just numpy cleaning
            cleaned = clean_numpy_types(suggestion)
            # Test if it's JSON serializable
            import json

            json.dumps(cleaned)
            return cleaned
        except (TypeError, ValueError):
            # If that fails, do a deep conversion to basic Python types
            return self._deep_clean(suggestion)

    def _deep_clean(self, obj):
        """Recursively convert any object to JSON-serializable Python types."""
        import json

        if isinstance(obj, dict):
            # Already a regular dict, just recursively clean values
            return {k: self._deep_clean(v) for k, v in obj.items()}
        elif hasattr(obj, "items"):
            # Handle dict-like objects (including WandB SummarySubDict)
            try:
                # Convert to regular dict first, then recursively clean
                return {k: self._deep_clean(v) for k, v in dict(obj).items()}
            except Exception:
                # If conversion fails, try JSON serialization as fallback
                try:
                    return json.loads(json.dumps(obj, default=str))
                except Exception:
                    return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._deep_clean(v) for v in obj]
        else:
            # For any other type, use clean_numpy_types first
            cleaned = clean_numpy_types(obj)
            # Then verify it's serializable
            try:
                json.dumps(cleaned)
                return cleaned
            except (TypeError, ValueError):
                return str(cleaned)

    def _load_runs(self):
        try:
            runs = self._get_runs_from_wandb()
            for run in runs:
                self._update_protein_from_run(run)

        except Exception as e:
            logger.warning(f"Could not load previous runs: {e}")

    def _get_runs_from_wandb(self) -> list:
        # Skip loading runs if this is not a real sweep or in offline mode
        if not self._sweep_id:
            return []

        # Add retry logic for network calls
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                logger.info(f"Starting API call to fetch runs for sweep {self._sweep_id}")
                logger.info(f"Project path: {self._wandb_run.entity}/{self._wandb_run.project}")

                # Try a simpler filter first to see if that's the issue
                # Just filter by sweep ID initially
                runs = self._api.runs(
                    path=f"{self._wandb_run.entity}/{self._wandb_run.project}",
                    filters={
                        "sweep": self._sweep_id,
                        # Temporarily comment out to test if this is causing slowness
                        # "summary_metrics.protein.state": {"$exists": True},
                        # "id": {"$ne": self._wandb_run.id},
                    },
                    order="+created_at",
                )

                api_call_time = time.time() - start_time
                logger.info(f"API call completed in {api_call_time:.2f} seconds")

                # OPTIMIZATION: Process runs lazily and limit the number fetched
                # This avoids fetching thousands of runs at once
                result_runs = []
                max_runs = self._max_runs_to_load

                logger.info(f"Fetching up to {max_runs} previous runs from sweep {self._sweep_id}")

                fetch_start = time.time()
                for _i, run in enumerate(runs):
                    # Apply filters manually after fetching
                    if run.id == self._wandb_run.id:
                        logger.debug(f"Skipping current run {run.id}")
                        continue

                    if not run.summary.get("protein.state"):
                        logger.debug(f"Skipping run {run.id} - no protein state")
                        continue

                    if len(result_runs) >= max_runs:
                        logger.info(f"Reached maximum of {max_runs} runs, stopping fetch")
                        break

                    result_runs.append(run)

                    # Log progress every 10 runs to show it's working
                    if (len(result_runs)) % 10 == 0:
                        logger.info(f"Fetched {len(result_runs)} runs...")

                fetch_time = time.time() - fetch_start
                logger.info(f"Successfully fetched {len(result_runs)} runs in {fetch_time:.2f} seconds")
                logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
                return result_runs

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.warning(f"Exception type: {type(e).__name__}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to fetch runs after {max_retries} attempts")
                    return []

    def _update_protein_from_run(self, run):
        logger.info(f"Processing run {run.name} (ID: {run.id})")

        if run.summary.get("protein.state") == "initializing":
            logger.info(f"Skipping run {run.name} - still initializing")
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
                self._defunct += 1
                logger.info(f"Skipping run {run.name} - defunct (no heartbeat for >5 min)")
                return
            self._num_running += 1
            logger.info(f"Skipping run {run.name} - still running")
            return

        try:
            suggestion, info = self._suggestion_from_run(run)
            logger.info(f"Extracted suggestion from run {run.name}: {suggestion}")
        except Exception as e:
            logger.warning(f"Failed to extract suggestion from run {run.name}: {e}")
            self._invalid += 1
            return

        # Check for duplicate "running" state check
        if run.summary.get("protein.state") == "running":
            self._num_running += 1
            return

        objective = run.summary.get("protein.objective", 0)
        cost = run.summary.get("protein.cost", 0)
        is_failure = run.summary.get("protein.state") == "failure"

        if is_failure:
            self._num_failures += 1
        else:
            self._num_observations += 1

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

        # CRITICAL FIX: Flatten the nested suggestion dict before passing to Protein
        # The suggestion from WandB is in nested format: {"trainer": {"optimizer": {"learning_rate": 0.0005}}}
        # But protein.observe() expects flattened format: {"trainer/optimizer/learning_rate": 0.0005}
        try:
            # Skip empty suggestions
            if not suggestion:
                logger.warning(f"Empty suggestion from run {run.name}, skipping observation")
                self._invalid += 1
                return

            # Check suggestion type before flattening
            if not isinstance(suggestion, dict):
                # Handle WandB's SummarySubDict and other dict-like objects
                try:
                    suggestion = dict(suggestion)
                except Exception as e:
                    logger.warning(f"Failed to convert suggestion from run {run.name} to dict: {e}")
                    logger.debug(f"Suggestion type: {type(suggestion)}")
                    self._invalid += 1
                    return

            logger.info(f"Original suggestion structure: {suggestion}")
            logger.info(f"Suggestion type: {type(suggestion)}, is dict: {isinstance(suggestion, dict)}")

            # Ensure we have a proper dict before flattening
            if hasattr(suggestion, "items"):
                # Convert dict-like objects to regular dict
                suggestion_dict = dict(suggestion)
                logger.info(f"Converted to dict: {suggestion_dict}")
            else:
                suggestion_dict = suggestion

            flattened_suggestion = dict(pufferlib.unroll_nested_dict(suggestion_dict))
            logger.info(f"Flattened suggestion: {flattened_suggestion}")

            # Validate flattened suggestion is not empty
            if not flattened_suggestion:
                logger.warning(f"Flattened suggestion from run {run.name} is empty. Original: {suggestion}")
                self._invalid += 1
                return

            # Pass flattened suggestion to Protein
            self._protein.observe(flattened_suggestion, objective, cost, is_failure)
            logger.info(f"Successfully recorded observation from run {run.name}: objective={objective}, cost={cost}")

        except AssertionError as e:
            # Handle missing parameters gracefully
            if "Missing hyperparameter" in str(e):
                logger.warning(f"Run {run.name} has incomplete parameters: {e}")
                logger.debug(f"Available parameters: {list(flattened_suggestion.keys())}")
                try:
                    logger.debug(f"Expected parameters: {list(self._protein.hyperparameters.flat_spaces.keys())}")
                except (AttributeError, TypeError):
                    logger.debug("Expected parameters: <not available>")
                # Skip this observation as it's incompatible with current config
                self._invalid += 1
            else:
                # Re-raise other assertion errors
                raise
        except KeyError as e:
            # Handle missing keys in hyperparameter spaces
            logger.warning(f"Run {run.name} has parameter not in current config: {e}")
            logger.debug(f"Flattened suggestion keys: {list(flattened_suggestion.keys())}")
            try:
                logger.debug(f"Expected parameters: {list(self._protein.hyperparameters.flat_spaces.keys())}")
            except (AttributeError, TypeError):
                logger.debug("Expected parameters: <not available>")
            self._invalid += 1
        except Exception as e:
            logger.warning(f"Failed to record observation in Protein for run {run.name}: {type(e).__name__}: {e}")
            logger.debug(f"Suggestion that failed: {suggestion}")
            logger.debug(f"Flattened suggestion: {flattened_suggestion}")
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Exception details: {str(e)}")
            # Try to provide more context about the failure
            try:
                logger.debug(f"Objective: {objective}, Cost: {cost}, Is_failure: {is_failure}")
            except Exception:
                pass
            # Still increment invalid count since we couldn't learn from this observation
            self._invalid += 1

        # Clean the info to remove any WandB objects before storing
        self._suggestion_info = self._deep_clean(info)

    def _suggestion_from_run(self, run):
        """
        Extract parameters from a run config - just retrieve what we stored.
        """
        # Try to get the flattened suggestion items first (newest format)
        suggestion_items = run.summary.get("protein.suggestion_flattened_items", None)

        if suggestion_items is not None and isinstance(suggestion_items, list):
            # Reconstruct the flattened dict from the list of tuples
            suggestion = dict(suggestion_items)
            logger.info(f"Reconstructed flattened suggestion from items for run {run.name}")
        else:
            # Try the old flattened format (which WandB auto-nests)
            suggestion = run.summary.get("protein.suggestion_flattened", None)

            if suggestion is not None:
                logger.info(f"Found flattened suggestion for run {run.name} (may be auto-nested by WandB)")
            else:
                # Fallback to nested format (oldest format)
                suggestion = run.summary.get("protein.suggestion", {})
                logger.info(f"Using nested suggestion format for run {run.name} (legacy)")

        # Handle case where WandB stored the suggestion as a string
        if isinstance(suggestion, str):
            try:
                # Try to parse the string back to a dictionary
                import ast

                suggestion = ast.literal_eval(suggestion)
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
                suggestion = {k: v for k, v in config_dict.items() if not k.startswith("_") and k != "dummy_param"}
            except Exception as e:
                logger.warning(f"Could not extract suggestion from config for run {run.name}: {e}")
                suggestion = {}

        # Get the stored prediction info
        info = run.summary.get("protein.suggestion_info", {})

        # Handle case where info might also be stored as a string
        if isinstance(info, str):
            try:
                import ast

                info = ast.literal_eval(info)
            except (ValueError, SyntaxError):
                info = {}

        # Ensure suggestion_uuid is set in info
        if "suggestion_uuid" not in info:
            info["suggestion_uuid"] = run.id

        return suggestion, info

    def _generate_protein_suggestion(self):
        """Generate a suggestion from Protein and store it."""
        # Always pass None as fill parameter to use Protein's default structure
        # The fill parameter expects a parameter structure, not metadata
        suggestion, info = self._protein.suggest(fill=None)

        # Clean numpy types from the suggestion before storing
        # This prevents numpy types from appearing in logs
        self._suggestion = clean_numpy_types(suggestion)
        self._suggestion_info = info

        # For WandB storage, we need to clean the suggestion
        # But we'll also store the flattened version explicitly
        cleaned_suggestion = self._deep_clean(self._suggestion)
        cleaned_info = self._deep_clean(info)

        # Save both nested and flattened versions to wandb summary
        # This ensures backward compatibility while fixing the loading issue
        # CRITICAL: WandB auto-nests keys with slashes, so we need to save the flattened
        # version as a list of (key, value) tuples to preserve the exact format
        flattened_items = list(self._suggestion.items()) if isinstance(self._suggestion, dict) else []

        self._wandb_run.summary.update(
            {
                "protein.suggestion": cleaned_suggestion,
                "protein.suggestion_flattened_items": flattened_items,  # Save as list to prevent auto-nesting
                "protein.suggestion_info": cleaned_info,
            }
        )


def create_wandb_sweep(sweep_name: str, wandb_entity: str, wandb_project: str):
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
