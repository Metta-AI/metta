"""Layer initialization utilities for PyTorch agents."""

import torch.nn as nn


def init_layer(layer, std=1.0):
    """
    Initialize layer weights to match ComponentPolicy initialization.

    Uses orthogonal initialization with configurable gain (std) and zeros for bias.
    This ensures consistent initialization between py_agent and YAML-based agents.

    Args:
        layer: PyTorch layer to initialize (must have weight parameter)
        std: Standard deviation/gain for orthogonal initialization (default: 1.0)

    Returns:
        The initialized layer
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer
