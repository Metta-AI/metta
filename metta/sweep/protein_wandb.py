import json
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Tuple, cast

import pufferlib
import wandb

from metta.common.util.numpy_helpers import clean_numpy_types
from metta.common.util.retry import retry_on_exception

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
        # Use provided run, else fall back to the currently active wandb run
        self._wandb_run = wandb_run or wandb.run
        assert self._wandb_run is not None, "No active wandb run found"
        self._wandb_run = cast(Any, self._wandb_run)

        # Derive the sweep ID, if this run belongs to a sweep.
        sweep_obj = getattr(self._wandb_run, "sweep", None)
        self._sweep_id = getattr(self._wandb_run, "sweep_id", None) or (sweep_obj.id if sweep_obj else None)

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

        # pyright: ignore[reportGeneralTypeIssues]
        assert self._wandb_run.summary.get("protein.state") is None, (
            f"Run {self._wandb_run.name} already has protein state"
        )

        self._wandb_run.summary.update({"protein.state": "initializing"})  # type: ignore[attr-defined]

        self._max_runs_to_load = max_runs_to_load

        self._load_runs()

        # Generate protein suggestion - let exceptions propagate
        self._generate_protein_suggestion()

        # Overwrite WandB agent's suggested parameters with Protein's suggestions
        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        # Unlock config to allow overwriting WandB agent's suggestions
        self._wandb_run.config.__dict__["_locked"] = {}  # type: ignore[attr-defined]

        # Set the actual parameter keys that training will use (overwriting WandB agent)
        self._wandb_run.config.update(wandb_config, allow_val_change=True)  # type: ignore[attr-defined]

        # Update state to "running" after initialization is complete
        self._wandb_run.summary.update({"protein.state": "running"})  # type: ignore[attr-defined]

    def record_observation(self, objective, cost):
        """
        Record an observation (objective, cost) for the current suggestion.

        Args:
            objective (float): The objective value to optimize (higher is better for maximization).
            cost (float): The cost of this evaluation (e.g., time taken).
        """
        # Update WandB with the observation
        self._wandb_run.summary.update(  # type: ignore[attr-defined]
            {
                "protein.objective": objective,
                "protein.cost": cost,
                "protein.state": "success",
            }
        )

        # Also record the observation with the Protein for future runs
        logger.info(f"Recording observation: {self._suggestion}, {objective}, {cost}")
        self._protein.observe(self._suggestion, objective, cost, False)

    def record_failure(self, error_message: str = "Unknown error"):
        """
        Record that the current suggestion failed.

        Args:
            error_message (str): Description of the failure.
        """
        # Update WandB with failure status
        self._wandb_run.summary.update(  # type: ignore[attr-defined]
            {
                "protein.state": "failure",
                "protein.error": error_message,
            }
        )

        # Also record the failure with the Protein
        # Non-Zero cost to avoid division/log errors
        self._protein.observe(self._suggestion, 0.0, 0.001, True)

    def suggest(self, fill=None) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Get the current suggestion.

        """
        # Return the already-cleaned suggestion and info
        return deepcopy(self._suggestion), deepcopy(self._suggestion_info)

    def _transform_suggestion(self, suggestion) -> dict[str, Any]:
        """Transform suggestion format. Override in subclasses if needed."""
        # Clean numpy types and any other non-serializable objects
        cleaned = clean_numpy_types(suggestion)

        # Test if it's JSON serializable
        # json.dumps(cleaned)
        return cleaned

    def _deep_clean(self, obj):
        """Recursively convert any object to JSON-serializable Python types."""
        if isinstance(obj, dict):
            # Already a regular dict, just recursively clean values
            return {k: self._deep_clean(v) for k, v in obj.items()}
        elif hasattr(obj, "items"):
            # Handle dict-like objects (including WandB SummarySubDict)
            # Convert to regular dict first, then recursively clean
            return {k: self._deep_clean(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [self._deep_clean(v) for v in obj]
        else:
            # For any other type, use clean_numpy_types first
            cleaned = clean_numpy_types(obj)
            # Then verify it's serializable
            json.dumps(cleaned)
            return cleaned

    def _load_runs(self):
        runs = self._get_runs_from_wandb()
        for run in runs:
            self._update_protein_from_run(run)

    @retry_on_exception(max_retries=3, retry_delay=2)
    def _get_runs_from_wandb(self) -> list:
        # Skip loading runs if this is not a real sweep or in offline mode
        if not self._sweep_id:
            return []

        start_time = time.time()
        logger.info(f"Starting API call to fetch runs for sweep {self._sweep_id}")
        logger.info(f"Project path: {self._wandb_run.entity}/{self._wandb_run.project}")

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
        for _, run in enumerate(runs):
            # Apply filters manually after fetching
            if run.id == self._wandb_run.id:
                continue

            if not run.summary.get("protein.state"):
                continue

            if len(result_runs) >= max_runs:
                logger.info(f"Reached maximum of {max_runs} runs, stopping fetch")
                break

            result_runs.append(run)

        fetch_time = time.time() - fetch_start
        logger.info(f"Successfully fetched {len(result_runs)} runs in {fetch_time:.2f} seconds")
        return result_runs

    def _validate_run(self, run):
        """
        Validate a run. Note that a failed run is not necessarily invalid.

        Args:
            run (wandb.Run): The run to validate.

        Returns:
            bool: True if the run is valid, False otherwise.
        """
        if run.summary.get("protein.state") == "initializing":
            logger.info(f"Skipping run {run.name} - still initializing")
            return False

        if run.summary.get("protein.state") == "running":
            # Parse heartbeat timestamp - WandB uses ISO format with or without microseconds
            heartbeat_str = run._attrs["heartbeatAt"]
            # Check if microseconds are present
            if "." in heartbeat_str:
                last_hb = datetime.strptime(heartbeat_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            else:
                last_hb = datetime.strptime(heartbeat_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
                self._defunct += 1
                logger.info(f"Skipping run {run.name} - defunct (no heartbeat for >5 min)")

            else:
                self._num_running += 1
                logger.info(f"Skipping run {run.name} - still running")

            return False

        return True

    def _get_suggestion_data_from_run(self, run):
        """
        Get the suggestion data from a run.

        Args:
            run (wandb.Run): The run to get the suggestion data from.

        Returns:
            tuple: A tuple containing the suggestion, info, objective, cost, and is_failure.
        """
        suggestion, info = self._suggestion_from_run(run)
        objective = run.summary.get("protein.objective", 0)
        cost = run.summary.get("protein.cost", 0)
        is_failure = run.summary.get("protein.state") == "failure"
        return suggestion, info, objective, cost, is_failure

    def _update_protein_from_run(self, run):
        if not self._validate_run(run):
            return

        suggestion, info, objective, cost, is_failure = self._get_suggestion_data_from_run(run)

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

        # Convert to dict if it's not already
        suggestion_dict = dict(suggestion)

        # Flatten the suggestion
        flattened_suggestion = dict(pufferlib.unroll_nested_dict(suggestion_dict))

        # Pass flattened suggestion to Protein
        self._protein.observe(flattened_suggestion, objective, cost, is_failure)

        # Clean the info to remove any WandB objects before storing
        cleaned_info = self._deep_clean(info)
        self._suggestion_info = cast(dict[str, Any], cleaned_info)

    def _suggestion_from_run(self, run):
        """
        Extract parameters from a run config - just retrieve what we stored.
        """
        # Get the flattened suggestion items (newest format)
        suggestion_items = run.summary.get("protein.suggestion_flattened_items")

        # Reconstruct the flattened dict from the list of tuples
        suggestion: dict[str, Any] = dict(suggestion_items) if suggestion_items else {}

        # If no flattened items, use nested suggestion stored directly in summary
        if not suggestion:
            summary_suggestion = run.summary.get("protein.suggestion", {})
            if summary_suggestion:
                # Convert to dict - this should already be a dict
                suggestion = dict(summary_suggestion)

        # Get the stored prediction info
        info = run.summary.get("protein.suggestion_info", {})

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
        self._suggestion_info = cast(dict[str, Any], cleaned_info)

        # Save both nested and flattened versions to wandb summary
        # CRITICAL: WandB auto-nests keys with slashes, so we need to save the flattened
        # version as a list of (key, value) tuples to preserve the exact format
        flattened_items = list(self._suggestion.items()) if isinstance(self._suggestion, dict) else []

        self._wandb_run.summary.update(  # type: ignore[attr-defined]
            {
                "protein.suggestion": cleaned_suggestion,
                "protein.suggestion_flattened_items": flattened_items,  # Save as list to prevent auto-nesting
                "protein.suggestion_info": cleaned_info,
            }
        )
