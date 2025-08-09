"""Protein optimization package for Metta."""

# Export core functions
from .protein import Protein
from .protein_metta import MettaProtein
from .protein_utils import convert_suggestion_to_cli_args, generate_protein_suggestion
from .sweep_lifecycle import evaluate_sweep_rollout, initialize_sweep, prepare_sweep_run
from .wandb_utils import (
    fetch_protein_observations_from_wandb,
    record_protein_observation_to_wandb,
)

__all__ = [
    "Protein",
    "MettaProtein",
    "generate_protein_suggestion",
    "convert_suggestion_to_cli_args",
    "fetch_protein_observations_from_wandb",
    "record_protein_observation_to_wandb",
    "prepare_sweep_run",
    "initialize_sweep",
    "evaluate_sweep_rollout",
]
