"""Hardware scaling experiment for adaptive hyperparameter optimization.

This experiment systematically explores how optimal hyperparameters and training
performance scale across different hardware configurations (GPU/node combinations).
"""

from .config import HardwareConfig, HardwareScalingConfig
from .scheduler import HardwareScalingScheduler

__all__ = [
    "HardwareConfig",
    "HardwareScalingConfig",
    "HardwareScalingScheduler",
]