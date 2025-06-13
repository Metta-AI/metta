import logging

from omegaconf import DictConfig, OmegaConf

from metta.common.util.numpy_helpers import clean_numpy_types

from .protein import Protein
from .protein_wandb import WandbProtein

logger = logging.getLogger("metta_protein")


class MettaProtein(WandbProtein):
    def __init__(
        self,
        cfg: DictConfig,
        wandb_run=None,
    ):
        # Convert parameters to container
        parameters = OmegaConf.to_container(cfg.parameters, resolve=True)

        # Create sweep_config with parameters and required fields
        sweep_config = {
            **parameters,
            "method": cfg.get("method", "bayes"),
            "metric": cfg.get("metric", "eval/mean_score"),
            "goal": cfg.get("goal", "maximize"),
        }

        # Convert protein config to container
        protein_cfg = OmegaConf.to_container(cfg.protein, resolve=True)

        # Initialize Protein with sweep_config as first arg and protein config as kwargs
        protein = Protein(
            sweep_config,
            **protein_cfg,
        )

        # Initialize WandbProtein with the created Protein instance
        super().__init__(protein, wandb_run)

    def _transform_suggestion(self, suggestion):
        """Transform suggestion format for compatibility with training."""
        return clean_numpy_types(suggestion)
