"""Muesli: Model-based Offline Policy Optimization.

A hybrid RL algorithm that combines model-free policy gradients with model-based
planning through learned dynamics models. Key features:
- CMPO (Clipped Maximum a Posteriori Policy Optimization)
- Multi-step model learning
- Retrace for off-policy correction
- Categorical value/reward representations
"""

from .agent import MuesliAgent
from .config import MuesliConfig
from .losses import compute_muesli_losses
from .replay_buffer import MuesliReplayBuffer
from .trainer import muesli_train

__all__ = [
    "MuesliAgent", 
    "MuesliConfig",
    "compute_muesli_losses",
    "MuesliReplayBuffer",
    "muesli_train"
]