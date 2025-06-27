"""
Doxascope Package

A module for training neural networks to predict agent movement from LSTM memory states.
"""

__all__ = [
    # data
    "DoxascopeLogger",
    "preprocess_doxascope_data",
    # network
    "DoxascopeNet",
    "DoxascopeTrainer",
    "prepare_data",
    # train
    "train_doxascope",
]

from .doxascope_data import DoxascopeLogger, preprocess_doxascope_data
from .doxascope_network import DoxascopeNet, DoxascopeTrainer, prepare_data
from .doxascope_train import train_doxascope
