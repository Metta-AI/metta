import logging

from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.numpy.clean_numpy_types import clean_numpy_types

from .protein import Protein
from .protein_wandb import WandbProtein

logger = logging.getLogger("metta_protein")


class MettaProtein(WandbProtein):
    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        wandb_run=None,
    ):
        parameters_dict = OmegaConf.to_container(cfg.parameters, resolve=True)

        protein = Protein(
            parameters_dict,
            cfg.protein.max_suggestion_cost,
            cfg.protein.resample_frequency,
            cfg.protein.num_random_samples,
            cfg.protein.global_search_scale,
            cfg.protein.random_suggestions,
            cfg.protein.suggestions_per_pareto,
        )

        # Initialize WandbProtein with the created Protein instance
        super().__init__(protein, wandb_run)

    def _transform_suggestion(self, suggestion):
        """Transform suggestion format for compatibility with training."""
        return clean_numpy_types(suggestion)
