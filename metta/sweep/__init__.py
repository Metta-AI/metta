"""Protein optimization package for Metta."""

# Export core functions
from .protein import Protein
from .protein_metta import MettaProtein
from .protein_utils import apply_protein_suggestion, generate_protein_suggestion
from .wandb_utils import (
    create_wandb_run_for_sweep,
    create_wandb_sweep,
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)

__all__ = [
    "Protein",
    "MettaProtein",
    "apply_protein_suggestion",
    "generate_protein_suggestion",
    "create_wandb_run_for_sweep",
    "create_wandb_sweep",
    "fetch_protein_observations_from_wandb",
    "record_protein_observation_to_wandb",
]
