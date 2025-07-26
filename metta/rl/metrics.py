"""General metrics functionality for Metta training."""

from typing import Any, Dict

import torch.nn as nn


def count_model_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: The PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: The PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of the model's parameters.

    Args:
        model: The PyTorch model

    Returns:
        Dictionary with model summary information
    """
    total_params = count_model_parameters(model)
    trainable_params = count_trainable_parameters(model)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }
