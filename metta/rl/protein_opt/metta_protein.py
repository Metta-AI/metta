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

        # Extract metadata from top level of sweep config
        metric = sweep_config.get("metric", "reward")
        goal = sweep_config.get("goal", "maximize")

        # Override num_random_samples if specified in sweep config
        if "num_random_samples" in sweep_config:
            num_random_samples = sweep_config["num_random_samples"]

        # Extract parameters from nested structure if present
        if "parameters" in sweep_config:
            # New structure: parameters are in a separate section
            parameters = sweep_config["parameters"]
        else:
            # Legacy structure: parameters at top level
            # Filter out metadata keys
            metadata_keys = {"metric", "goal", "num_random_samples", "protein"}
            parameters = {k: v for k, v in sweep_config.items() if k not in metadata_keys}

        # Create clean config for Protein with parameters + metadata
        clean_config = dict(parameters)  # Copy actual parameters only
        clean_config["metric"] = metric
        clean_config["goal"] = goal

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
        import numpy as np

        def clean_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.item() if obj.size == 1 else obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_numpy_types(v) for v in obj]
            return obj

        # Convert numpy types to Python native types for OmegaConf compatibility
        return clean_numpy_types(suggestion)
