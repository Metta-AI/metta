import json
import logging
import math
import random
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import List, Set

import wandb
from carbs import (
    CARBS,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    Param,
    SuggestionInBasic,
)

logger = logging.getLogger("wandb_carbs")
# logger.setLevel(logging.DEBUG)


class WandbCarbs:
    def __init__(self, carbs: CARBS, wandb_run=None, sweep_id: str = None):
        """
        Initialize WandbCarbs with a CARBS instance and optionally a wandb run.

        Args:
            carbs (CARBS): The CARBS instance to use for suggestions.
            wandb_run (wandb.Run, optional): The wandb run to use. If None, uses the current run.
        """
        logger.warning("WandbCarbs is deprecated, use MettaProtein instead")
        self._wandb_run = wandb_run or wandb.run
        self._sweep_id = self._wandb_run.sweep_id
        self._api = wandb.Api()

        self._carbs = carbs
        self._carbs._set_seed(int(time.time()))
        self._num_observations = 0
        self._num_failures = 0
        self._num_running = 0
        self._defunct = 0
        self._invalid = 0
        self._observations = []

        assert self._wandb_run.summary.get("carbs.state") is None, f"Run {self._wandb_run.name} already has carbs state"

        self._wandb_run.summary.update({"carbs.state": "initializing"})
        self._load_runs()
        self._generate_carbs_suggestion()

        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        del wandb_config["suggestion_uuid"]
        self._wandb_run.config.__dict__["_locked"] = {}
        self._wandb_run.config.update(wandb_config, allow_val_change=True)
        self._wandb_run.summary.update({"carbs.state": "running"})

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
            assert wandb_run.summary["carbs.state"] == "running", (
                f"Run is not running, cannot record observation {wandb_run.summary}"
            )

        wandb_run.summary.update({"carbs.objective": objective, "carbs.cost": cost, "carbs.state": "success"})
        logger.info(f"Recording observation ({objective}, {cost}) for {wandb_run.name}")

    def record_failure(self):
        self._record_failure(self._wandb_run)

    @staticmethod
    def _record_failure(wandb_run):
        """
        Record a failure for the current run.
        """
        logger.info(f"Recording failure for {wandb_run.name}")
        wandb_run.summary.update({"carbs.state": "failure"})

    def suggest(self):
        """
        Get the current suggestion.

        Returns:
            dict: The current suggestion.
        """
        return self._transform_suggestion(deepcopy(self._suggestion))

    def _transform_suggestion(self, suggestion):
        return suggestion

    def _load_runs(self):
        logger.info(f"Loading previous runs from sweep {self._sweep_id}")

        for run in self._get_runs_from_wandb():
            self._update_carbs_from_run(run)

        logger.info(
            "Initialized CARBS with "
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
        runs = self._api.runs(
            path=f"{self._wandb_run.entity}/{self._wandb_run.project}",
            filters={
                "sweep": self._sweep_id,
                # "tags": {"$in": [f"sweep_id:{self._sweep_id}"]},
                "summary_metrics.carbs.state": {"$exists": True},
                "id": {"$ne": self._wandb_run.id},
            },
            order="+created_at",
        )
        return runs

    def _update_carbs_from_run(self, run):
        if run.summary["carbs.state"] == "initializing":
            logger.debug(f"skipping run {run.name} because it is initializing")

        if run.summary["carbs.state"] == "running":
            last_hb = datetime.strptime(run._attrs["heartbeatAt"], "%Y-%m-%dT%H:%M:%S%fZ").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
                logger.debug(f"skipping run {run.name} because it has not heartbeated in the last 5 minutes")
                self._defunct += 1
                return

        try:
            suggestion = self._suggestion_from_run(run)
        except Exception as e:
            logger.warning(f"Failed to get suggestion from run {run.name}: {e}")
            self._invalid += 1
            return

        self._carbs._remember_suggestion(
            suggestion, SuggestionInBasic(self._carbs._param_space_real_to_basic_space_real(suggestion)), run.id
        )

        if run.summary["carbs.state"] == "running":
            logger.debug(f"recording suggestion run {run.name} that is still running")
            self._num_running += 1
            return

        objective = run.summary.get("carbs.objective", 0)
        cost = run.summary.get("carbs.cost", 0)

        if run.summary["carbs.state"] == "failure":
            self._num_failures += 1
        else:
            self._num_observations += 1

        logger.debug(
            f"Observation {run.name} "
            + f"{objective} / {cost} "
            + f"failure: {run.summary['carbs.state'] == 'failure'} "
            + json.dumps(suggestion, indent=2)
        )
        self._observations.append(
            {
                "suggestion": suggestion,
                "objective": objective,
                "cost": cost,
                "is_failure": run.summary["carbs.state"] == "failure",
                "run_id": run.id,
                "run_name": run.name,
            }
        )
        self._carbs.observe(
            ObservationInParam(
                input=suggestion, output=objective, cost=cost, is_failure=run.summary["carbs.state"] == "failure"
            )
        )

    def _suggestion_from_run(self, run):
        suggestion = {param.name: run.config.get(param.name, param.search_center) for param in self._carbs.params}
        suggestion["suggestion_uuid"] = run.id
        return suggestion

    def _generate_carbs_suggestion(self):
        while True:
            try:
                self._suggestion = self._carbs.suggest().suggestion
                break
            except Exception as e:
                logger.warning(f"Failed to suggest: {e}")
                logger.debug(traceback.format_exc())

                if len(self._carbs.success_observations) == 0 and len(self._carbs.failure_observations) == 0:
                    logger.error("Unable to recover from failed suggestion, no observations to remove")
                    raise e

                # Remove a random element from success and failure observations
                if len(self._carbs.success_observations):
                    logger.info("Removing random success observation")
                    self._carbs.success_observations.pop(random.randint(0, len(self._carbs.success_observations) - 1))
                elif len(self._carbs.failure_observations):
                    logger.info("Removing random failure observation")
                    self._carbs.failure_observations.pop(random.randint(0, len(self._carbs.failure_observations) - 1))


class Pow2WandbCarbs(WandbCarbs):
    """
    A subclass of WandbCarbs that handles parameters that should be treated as powers of 2.

    This class extends WandbCarbs to support parameters that are internally represented as
    exponents but should be presented as powers of 2 externally.

    Attributes:
        pow2_params (Set[str]): A set of parameter names that should be treated as powers of 2.

    """

    def __init__(self, carbs: CARBS, pow2_params: Set[str], wandb_run=None):
        """
        Initialize the Pow2WandbCarbs instance.

        Args:
            carbs (CARBS): The CARBS instance to use for optimization.
            pow2_params (Set[str]): A set of parameter names to be treated as powers of 2.
            wandb_run: The Weights & Biases run object (optional).
        """
        self.pow2_params = pow2_params or set()
        super().__init__(carbs, wandb_run)

    def _transform_suggestion(self, suggestion):
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                suggestion[param.name] = 2 ** suggestion[param.name]
        return suggestion

    def _suggestion_from_run(self, run):
        suggestion = super()._suggestion_from_run(run)
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                try:
                    suggestion[param.name] = int(math.log2(suggestion[param.name]))
                except ValueError as e:
                    logger.warning(f"Failed to convert {run.name}:{param.name} to power of 2: {suggestion[param.name]}")
                    raise e

        return suggestion


def create_sweep(sweep_name: str, wandb_entity: str, wandb_project: str, carb_params: List[Param]):
    """
    Create a new wandb sweep based on CARBS parameters.

    Args:
        sweep_name (str): The name of the sweep.
        wandb_entity (str): The wandb entity (username or team name).
        wandb_project (str): The wandb project name.
        carb_params (List[Param]): The CARBS parameter spaces.

    Returns:
        str: The ID of the created sweep.
    """
    sweep_id = wandb.sweep(
        sweep=_wandb_sweep_cfg_from_carbs_params(sweep_name, carb_params),
        project=wandb_project,
        entity=wandb_entity,
    )
    return sweep_id


def _wandb_sweep_cfg_from_carbs_params(name, carb_params: List[Param]):
    wandb_sweep_cfg = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "carbs.objective",
        },
        "parameters": {},
        "name": name,
    }
    for param in carb_params:
        wandb_sweep_cfg["parameters"][param.name] = {
            "min": param.space.min,
            "max": param.space.max,
            "distribution": _wandb_distribution(param),
        }
    return wandb_sweep_cfg


def _wandb_distribution(param: Param):
    if isinstance(param.space, LogSpace):
        return "log_uniform_values"
    elif isinstance(param.space, LogitSpace):
        return "uniform"
    elif isinstance(param.space, LinearSpace):
        if param.space.is_integer:
            return "int_uniform"
        else:
            return "uniform"


__all__ = [
    "WandbCarbs",
    "Pow2WandbCarbs",
    "create_sweep",
]
