"""Centralized layer initialization for PyTorch agents."""

import torch.nn as nn


def init_layer(layer, std=1.0):
    """Initialize layer weights to match ComponentPolicy initialization.

    Uses orthogonal initialization with configurable gain, matching the
    initialization used by YAML-configured agents through ComponentPolicy.

    Args:
        layer: The neural network layer to initialize
        std: The gain/standard deviation for initialization (default: 1.0)

    Returns:
        The initialized layer
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer
