import logging

from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.rl.protein import Protein
from metta.rl.protein_opt.wandb_protein import WandbProtein

logger = logging.getLogger("metta_protein")


class MettaProtein(WandbProtein):
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
        # Extract the sweep config dictionary - ensure we get just the sweep part
        if isinstance(cfg, (DictConfig, ListConfig)):
            if hasattr(cfg, "sweep") and cfg.sweep != "???":  # Check for unresolved value
                sweep_config = OmegaConf.to_container(cfg.sweep, resolve=True)
            else:
                sweep_config = OmegaConf.to_container(cfg, resolve=True)
        else:
            sweep_config = dict(cfg.get("sweep", cfg) if hasattr(cfg, "get") else cfg)

        # Handle case where sweep_config still has nested sweep structure
        if isinstance(sweep_config, dict) and "sweep" in sweep_config and isinstance(sweep_config["sweep"], dict):
            sweep_config = sweep_config["sweep"]

        if not isinstance(sweep_config, dict) or sweep_config is None:
            raise ValueError("Sweep config must be a dict.")

        # Add metric and goal if not present
        if isinstance(sweep_config, dict):
            if "metric" not in sweep_config:
                sweep_config["metric"] = "reward"
            if "goal" not in sweep_config:
                sweep_config["goal"] = "maximize"

        # Extract parameters and create a clean config for Protein
        if "parameters" in sweep_config:
            # Create a clean config with only parameter definitions + required metadata
            clean_config = dict(sweep_config["parameters"])
            clean_config["metric"] = sweep_config["metric"]
            clean_config["goal"] = sweep_config["goal"]
        else:
            # For flat parameter configs, create clean config with just parameters + metadata
            clean_config = {}
            for key, value in sweep_config.items():
                # Include parameters (dicts with distribution info) and required metadata
                if key in ("metric", "goal"):
                    clean_config[key] = value
                elif isinstance(value, dict) and "distribution" in value:
                    clean_config[key] = value
                # Skip other metadata fields like num_random_samples, method, etc.

        sweep_config = clean_config

        # Create Protein instance with cleaned config
        protein = Protein(
            sweep_config,
            max_suggestion_cost,
            resample_frequency,
            num_random_samples,
            global_search_scale,
            random_suggestions,
            suggestions_per_pareto,
        )

        # Initialize WandbProtein with the created Protein instance
        super().__init__(protein, wandb_run)

    def _transform_suggestion(self, suggestion):
        """Transform suggestion format for compatibility with training."""
        # Keep the suggestion as-is, no transformation needed for MettaProtein
        return suggestion
