from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.rl.protein import Protein


class MettaProtein(Protein):
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        max_suggestion_cost=3600,
        resample_frequency=0,
        num_random_samples=50,
        global_search_scale=1,
        random_suggestions=1024,
        suggestions_per_pareto=256,
    ):
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

    # def _get_runs_from_wandb(self):
    #     pass

    # def _expand_search_space(self, fill):
    #     pass

    # def _seed_with_search_center(self, fill):
    #     pass

    # def _suggest_random(self, fill):
    #     pass
