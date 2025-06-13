"""
Doxascope Network Module

This module contains the neural network architecture and training code
for the doxascope system that predicts agent movement from LSTM memory vectors.

Key components:
- DoxascopeNet: Optimized neural network architecture
- DoxascopeTrainer: Training and evaluation infrastructure
- Parameter sweep scripts for hyperparameter optimization
"""

from .doxascope_network import DoxascopeNet, DoxascopeTrainer, train_doxascope

__all__ = ["DoxascopeNet", "DoxascopeTrainer", "train_doxascope"]
