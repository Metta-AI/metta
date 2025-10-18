"""
Doxascope Package

A module for training neural networks to predict agent relative positions from LSTM memory states.
"""

__all__ = [
    # data
    "DoxascopeLogger",
    "preprocess_doxascope_data",
    # network
    "DoxascopeNet",
    "DoxascopeTrainer",
    "TrainingResult",
    "prepare_data",
]

from .doxascope_data import DoxascopeLogger, preprocess_doxascope_data
from .doxascope_network import DoxascopeNet, DoxascopeTrainer, TrainingResult, prepare_data
