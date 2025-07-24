"""Protein optimization package for Metta."""

# Export lifecycle functions
from .lifecycle.sweep_eval import main as sweep_eval_main
from .lifecycle.sweep_prepare_run import setup_next_run
from .lifecycle.sweep_setup import create_sweep
from .protein import Protein
from .protein_metta import MettaProtein

__all__ = [
    "Protein",
    "MettaProtein",
    "sweep_eval_main",
    "setup_next_run",
    "create_sweep",
]
