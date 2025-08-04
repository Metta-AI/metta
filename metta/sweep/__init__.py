"""Protein optimization package for Metta."""

import sys
from typing import TYPE_CHECKING, Any

# For type checking, import everything normally
if TYPE_CHECKING:
    from .protein import Protein
    from .protein_metta import MettaProtein
    from .sweep_lifecycle import evaluate_sweep_rollout, initialize_sweep, prepare_sweep_run
    from .wandb_utils import (
        fetch_protein_observations_from_wandb,
        record_protein_observation_to_wandb,
    )

__all__ = [
    "Protein",
    "MettaProtein",
    "fetch_protein_observations_from_wandb",
    "record_protein_observation_to_wandb",
    "prepare_sweep_run",
    "initialize_sweep",
    "evaluate_sweep_rollout",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism to defer loading modules until they're actually used.

    This prevents heavy dependencies like torch/pyro from being loaded when only
    lightweight utilities are needed.
    """
    # Map of attribute names to their module and actual name
    import_map = {
        "Protein": ("protein", "Protein"),
        "MettaProtein": ("protein_metta", "MettaProtein"),
        "fetch_protein_observations_from_wandb": ("wandb_utils", "fetch_protein_observations_from_wandb"),
        "record_protein_observation_to_wandb": ("wandb_utils", "record_protein_observation_to_wandb"),
        "prepare_sweep_run": ("sweep_lifecycle", "prepare_sweep_run"),
        "initialize_sweep": ("sweep_lifecycle", "initialize_sweep"),
        "evaluate_sweep_rollout": ("sweep_lifecycle", "evaluate_sweep_rollout"),
    }

    if name in import_map:
        module_name, attr_name = import_map[name]
        # Import the module
        from importlib import import_module

        module = import_module(f".{module_name}", package=__name__)
        # Get the attribute
        attr = getattr(module, attr_name)
        # Cache it in the module for future use
        setattr(sys.modules[__name__], name, attr)
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
