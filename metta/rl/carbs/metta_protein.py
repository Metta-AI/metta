from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.rl.protein import Protein


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
        # Store wandb_run for compatibility
        self.wandb_run = wandb_run
        self.cfg = cfg

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

        super().__init__(
            sweep_config,
            max_suggestion_cost,
            resample_frequency,
            num_random_samples,
            global_search_scale,
            random_suggestions,
            suggestions_per_pareto,
        )

        # Load previous runs from WandB (stubbed for now)
        self._load_runs_from_wandb()

    def suggest(self, fill=None):
        """Override suggest to handle optional fill parameter and return format compatibility."""
        result, info = super().suggest(fill)
        return result

    def _load_runs_from_wandb(self):
        """Stub for loading previous runs from WandB."""
        # TODO: Implement WandB integration to load previous observations
        # This would load previous runs and call self.observe() for each
        if self.wandb_run:
            print(f"[STUB] Would load previous runs from WandB sweep for {self.wandb_run.name}")
        pass

    @staticmethod
    def _record_observation(wandb_run, objective: float, cost: float, allow_update: bool = False):
        """Stub for recording observation to WandB (compatibility with sweep_eval.py)."""
        # TODO: Implement WandB observation recording
        if wandb_run:
            print(f"[STUB] Would record observation: score={objective}, cost={cost} to {wandb_run.name}")
        pass

    def record_failure(self):
        """Stub for recording failure (compatibility with existing code)."""
        # TODO: Implement failure recording
        if self.wandb_run:
            print(f"[STUB] Would record failure for {self.wandb_run.name}")
        pass
