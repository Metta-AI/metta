import json
import logging
from datetime import datetime, timezone

import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.rl.protein import Protein

logger = logging.getLogger("metta_protein")


class MettaProtein(Protein):
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        wandb_run=None,
        max_suggestion_cost=3600,
        resample_frequency=0,
        num_random_samples=50,
        global_search_scale=1,
        random_suggestions=1024,
        suggestions_per_pareto=256,
    ):
        # Store wandb_run and initialize tracking
        self.wandb_run = wandb_run or wandb.run
        self.cfg = cfg
        self._num_observations = 0
        self._num_failures = 0
        self._num_running = 0
        self._defunct = 0
        self._invalid = 0

        # Extract the sweep config dictionary
        if isinstance(cfg, (DictConfig, ListConfig)):
            base = cfg.sweep if hasattr(cfg, "sweep") else cfg
            sweep_config = OmegaConf.to_container(base, resolve=True)
        else:
            sweep_config = dict(cfg)
        if not isinstance(sweep_config, dict) or sweep_config is None:
            raise ValueError("Sweep config must be a dict.")

        # Add metric and goal if not present
        if isinstance(sweep_config, dict):
            if "metric" not in sweep_config:
                sweep_config["metric"] = "reward"
            if "goal" not in sweep_config:
                sweep_config["goal"] = "maximize"

        # Extract parameters from sweep config
        if "parameters" in sweep_config:
            sweep_config = {
                **sweep_config["parameters"],
                "metric": sweep_config["metric"],
                "goal": sweep_config["goal"],
            }

        super().__init__(
            sweep_config,
            max_suggestion_cost,
            resample_frequency,
            num_random_samples,
            global_search_scale,
            random_suggestions,
            suggestions_per_pareto,
        )

        if self.wandb_run:
            # Initialize WandB state
            assert self.wandb_run.summary.get("protein.state") is None, (
                f"Run {self.wandb_run.name} already has protein state"
            )
            self.wandb_run.summary.update({"protein.state": "initializing"})

            # Load previous runs and update state
            self._load_runs_from_wandb()

            # Generate and record initial suggestion
            suggestion, _ = self.suggest()
            wandb_config = dict(suggestion)
            if "suggestion_uuid" in wandb_config:
                del wandb_config["suggestion_uuid"]
            self.wandb_run.config.__dict__["_locked"] = {}
            self.wandb_run.config.update({"parameters": wandb_config}, allow_val_change=True)
            self.wandb_run.summary.update({"protein.state": "running"})

    def suggest(self, fill=None):
        """Override suggest to handle optional fill parameter and return format compatibility."""
        result, info = super().suggest(fill)
        if self.wandb_run and "suggestion_uuid" not in result:
            result["suggestion_uuid"] = self.wandb_run.id
        return result, info

    def _load_runs_from_wandb(self):
        """Load previous runs from WandB and update optimizer state."""
        if not self.wandb_run:
            return

        logger.info(f"Loading previous runs from sweep {self.wandb_run.sweep_id}")
        api = wandb.Api()

        # Query runs from this sweep
        runs = api.runs(
            path=f"{self.wandb_run.entity}/{self.wandb_run.project}",
            filters={
                "sweep": self.wandb_run.sweep_id,
                "summary_metrics.protein.state": {"$exists": True},
                "id": {"$ne": self.wandb_run.id},
            },
            order="+created_at",
        )

        # Process each run
        for run in runs:
            if run.summary["protein.state"] == "initializing":
                logger.debug(f"Skipping run {run.name} because it is initializing")
                continue

            if run.summary["protein.state"] == "running":
                last_hb = datetime.strptime(run._attrs["heartbeatAt"], "%Y-%m-%dT%H:%M:%S%fZ").replace(
                    tzinfo=timezone.utc
                )
                if (datetime.now(timezone.utc) - last_hb).total_seconds() > 5 * 60:
                    logger.debug(f"Skipping run {run.name} - no heartbeat in last 5 minutes")
                    self._defunct += 1
                    continue
                self._num_running += 1
                continue

            # Extract parameters and results
            try:
                params = {k: v for k, v in run.config.items() if not k.startswith("_")}
                if "parameters" in params:
                    params = params["parameters"]

                objective = run.summary.get("protein.objective", 0)
                cost = run.summary.get("protein.cost", 0)
                is_failure = run.summary["protein.state"] == "failure"

                # Use the base class observe method to properly add observations
                # Note: we don't pass suggestion_uuid to observe since it's not part of the hyperparameter space
                self.observe(params, objective, cost, is_failure)

                if is_failure:
                    self._num_failures += 1
                else:
                    self._num_observations += 1

            except Exception as e:
                logger.warning(f"Failed to process run {run.name}: {e}")
                self._invalid += 1

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

    @staticmethod
    def _record_observation(wandb_run, objective: float, cost: float, allow_update: bool = False):
        """Record an observation to WandB."""
        if not wandb_run:
            return

        if not allow_update:
            assert wandb_run.summary["protein.state"] == "running", (
                f"Run is not running, cannot record observation {wandb_run.summary}"
            )

        wandb_run.summary.update({"protein.objective": objective, "protein.cost": cost, "protein.state": "success"})
        logger.info(f"Recording observation ({objective}, {cost}) for {wandb_run.name}")

    def record_failure(self):
        """Record a failure to WandB."""
        if not self.wandb_run:
            return

        logger.info(f"Recording failure for {self.wandb_run.name}")
        self.wandb_run.summary.update({"protein.state": "failure"})
