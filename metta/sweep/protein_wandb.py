import json
import logging
from copy import deepcopy
from datetime import datetime, timezone

import wandb

from metta.common.util.wandb.wandb_context import WandbRun

from .protein import Protein

logger = logging.getLogger("wandb_protein")


class WandbProtein:
    def __init__(
        self,
        protein: Protein,
        wandb_run: WandbRun | None = None,
    ):
        """
        Initialize WandbProtein with a Protein instance and optionally a wandb run.

        Args:
            protein (Protein): The Protein instance to use for suggestions.
            wandb_run (wandb.Run, optional): The wandb run to use. If None, uses the current run.
        """
        _wandb_run = wandb_run or wandb.run
        assert _wandb_run is not None, "No active wandb run found"
        self._wandb_run = _wandb_run

        self._sweep_id = self._wandb_run.sweep_id
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
        self._load_runs()
        self._generate_protein_suggestion()

        # CRITICAL: Overwrite WandB agent's suggested parameters with Protein's suggestions
        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        # Unlock config to allow overwriting WandB agent's suggestions
        self._wandb_run.config.__dict__["_locked"] = {}

        # Set the actual parameter keys that training will use (overwriting WandB agent)
        self._wandb_run.config.update(wandb_config, allow_val_change=True)

        # Also store in parameters section for tracking
        self._wandb_run.config.update({"parameters": wandb_config}, allow_val_change=True)

        # Store prediction info in wandb summary
        if self._suggestion_info:
            self._wandb_run.summary.update(
                {
                    "protein.prediction.cost": self._suggestion_info.get("cost"),
                    "protein.prediction.score": self._suggestion_info.get("score"),
                    "protein.prediction.rating": self._suggestion_info.get("rating"),
                }
            )

        # Store the complete suggestion for easy retrieval
        self._wandb_run.summary.update({"protein.suggestion": self._suggestion})
        self._wandb_run.summary.update({"protein.suggestion_info": self._suggestion_info})
        self._wandb_run.summary.update({"protein.state": "running"})

    def record_observation(self, objective: float, cost: float, allow_update: bool = False):
        self._record_observation(self._wandb_run, objective, cost, allow_update)

    @staticmethod
    def _record_observation(wandb_run, objective: float, cost: float, allow_update: bool = False):
        """
        Record an observation for the current run.

        Args:
            objective (float): The objective value to record.
            cost (float): The cost value to record.
            allow_update (bool, optional): If True, allows updating even if the run is not in "running" state.
        """
        if not allow_update:
            assert wandb_run.summary["protein.state"] == "running", (
                f"Run is not running, cannot record observation {wandb_run.summary}"
            )

        wandb_run.summary.update({"protein.objective": objective, "protein.cost": cost, "protein.state": "success"})
        logger.info(f"Recording observation ({objective}, {cost}) for {wandb_run.name}")

    def record_failure(self):
        self._record_failure(self._wandb_run)

    @staticmethod
    def _record_failure(wandb_run):
        """
        Record a failure for the current run.
        """
        logger.info(f"Recording failure for {wandb_run.name}")
        wandb_run.summary.update({"protein.state": "failure"})

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
            for run in self._get_runs_from_wandb():
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

        runs = self._api.runs(
            path=f"{self._wandb_run.entity}/{self._wandb_run.project}",
            filters={
                "sweep": self._sweep_id,
                "summary_metrics.protein.state": {"$exists": True},
                "id": {"$ne": self._wandb_run.id},
            },
            order="+created_at",
        )
        return runs

    def _update_protein_from_run(self, run):
        if run.summary["protein.state"] == "initializing":
            logger.debug(f"Skipping run {run.name} because it is initializing")
            return

        if run.summary["protein.state"] == "running":
            last_hb = datetime.strptime(run._attrs["heartbeatAt"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
                logger.debug(f"Skipping run {run.name} - no heartbeat in last 5 minutes")
                self._defunct += 1
                return
            self._num_running += 1
            return

        try:
            suggestion, info = self._suggestion_from_run(run)
        except Exception as e:
            logger.warning(f"Failed to get suggestion from run {run.name}: {e}")
            self._invalid += 1
            return

        if run.summary["protein.state"] == "running":
            logger.debug(f"Recording suggestion run {run.name} that is still running")
            self._num_running += 1
            return

        objective = run.summary.get("protein.objective", 0)
        cost = run.summary.get("protein.cost", 0)
        is_failure = run.summary["protein.state"] == "failure"

        if is_failure:
            self._num_failures += 1
        else:
            self._num_observations += 1

        logger.debug(
            f"Observation {run.name} "
            + f"{objective} / {cost} "
            + f"failure: {is_failure} "
            + json.dumps(suggestion, indent=2)
        )

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

        # Pass suggestion to Protein (no need to remove suggestion_uuid since it's now in info)
        self._protein.observe(suggestion, objective, cost, is_failure)
        self._suggestion_info = info

    def _suggestion_from_run(self, run):
        """
        Extract parameters from a run config - just retrieve what we stored.
        """
        # Get the stored suggestion
        suggestion = run.summary.get("protein.suggestion", {})

        # Get the stored prediction info
        info = run.summary.get("protein.suggestion_info", {})

        # Ensure suggestion_uuid is set in info
        if "suggestion_uuid" not in info:
            info["suggestion_uuid"] = run.id

        return suggestion, info

    def _generate_protein_suggestion(self):
        """Generate a suggestion from Protein optimizer."""
        try:
            self._suggestion, self._suggestion_info = self._protein.suggest(fill=None)

            # Add suggestion UUID to info (metadata)
            if "suggestion_uuid" not in self._suggestion_info:
                self._suggestion_info["suggestion_uuid"] = self._wandb_run.id

            # Update wandb summary with the new suggestion and info
            self._wandb_run.summary.update({"protein.suggestion": self._suggestion})
            self._wandb_run.summary.update({"protein.suggestion_info": self._suggestion_info})

        except Exception as e:
            logger.error(f"Failed to generate Protein suggestion: {e}")
            raise e


def create_sweep(sweep_name: str, wandb_entity: str, wandb_project: str):
    """
    Create a new wandb sweep without parameters (Protein will control all suggestions).

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
            "parameters": {},  # Empty - Protein controls all parameters
        },
        project=wandb_project,
        entity=wandb_entity,
    )
    return sweep_id
