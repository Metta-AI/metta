import logging

from omegaconf import DictConfig, ListConfig, OmegaConf

from .protein import Protein
from .protein_wandb import WandbProtein

logger = logging.getLogger("metta_protein")


class MettaProtein(WandbProtein):
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        wandb_run=None,
    ):
        # Get parameters section or empty dict
        parameters = cfg.sweep.get("parameters", {})

        # Convert to container if it's an OmegaConf object, otherwise use as-is
        if OmegaConf.is_config(parameters):
            parameters_dict = OmegaConf.to_container(parameters, resolve=True)
        else:
            parameters_dict = parameters

        protein = Protein(
            parameters_dict,
            cfg.sweep.protein.get("max_suggestion_cost", 3600),
            cfg.sweep.protein.get("resample_frequency", 0),
            cfg.sweep.protein.get("num_random_samples", 50),
            cfg.sweep.protein.get("global_search_scale", 1),
            cfg.sweep.protein.get("random_suggestions", 1024),
            cfg.sweep.protein.get("suggestions_per_pareto", 256),
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
