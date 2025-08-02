"""Categorical representation utilities for Muesli.

Implements the two-hot encoding scheme used in MuZero/Muesli for representing
scalar values as categorical distributions over a fixed support.
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def scalar_to_support(
    x: Tensor,
    support_size: int = 601,
    value_min: float = -300.0,
    value_max: float = 300.0,
    epsilon: float = 0.001
) -> Tensor:
    """Convert scalar values to categorical distributions using two-hot encoding.
    
    This provides better gradient flow than one-hot encoding by distributing
    the scalar value between the two nearest support bins.
    
    Args:
        x: Scalar values to convert (shape: [...])
        support_size: Number of bins in the support
        value_min: Minimum value of the support range
        value_max: Maximum value of the support range
        epsilon: Small value to avoid numerical issues
        
    Returns:
        Categorical distributions (shape: [..., support_size])
    """
    # Create support bins
    support = torch.linspace(value_min, value_max, support_size, device=x.device)
    
    # Clamp values to support range
    x = torch.clamp(x, value_min, value_max)
    
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(-1)  # [..., 1]
    support_expanded = support.view(1, -1).expand(*x.shape, -1)  # [..., support_size]
    
    # Find the two nearest bins
    # below: index of the largest support value <= x
    below_mask = support_expanded <= x_expanded
    below_indices = below_mask.sum(dim=-1) - 1  # [...] 
    below_indices = torch.clamp(below_indices, 0, support_size - 2)
    
    above_indices = below_indices + 1
    
    # Get the support values at these indices
    # We need to gather along the last dimension
    batch_shape = x.shape
    flat_below = below_indices.flatten()
    flat_above = above_indices.flatten()
    
    # Create indices for gathering
    gather_indices = torch.arange(flat_below.shape[0], device=x.device)
    
    # Reshape support for gathering
    support_flat = support_expanded.reshape(-1, support_size)
    
    below_values = support_flat[gather_indices, flat_below].reshape(batch_shape)
    above_values = support_flat[gather_indices, flat_above].reshape(batch_shape)
    
    # Calculate weights (distance-based interpolation)
    # When x equals below_values, below_weight should be 1
    # When x equals above_values, below_weight should be 0
    denom = above_values - below_values + epsilon
    below_weight = (above_values - x) / denom
    above_weight = (x - below_values) / denom
    
    # Create categorical distribution
    categorical = torch.zeros(*x.shape, support_size, device=x.device)
    
    # Flatten for indexing
    categorical_flat = categorical.reshape(-1, support_size)
    
    # Set the weights
    categorical_flat[gather_indices, flat_below] = below_weight.flatten()
    categorical_flat[gather_indices, flat_above] = above_weight.flatten()
    
    return categorical.reshape(*x.shape, support_size)


def support_to_scalar(
    categorical: Tensor,
    support_size: int = 601,
    value_min: float = -300.0,
    value_max: float = 300.0
) -> Tensor:
    """Convert categorical distributions back to scalar values.
    
    Args:
        categorical: Categorical distributions (shape: [..., support_size])
        support_size: Number of bins in the support
        value_min: Minimum value of the support range
        value_max: Maximum value of the support range
        
    Returns:
        Scalar values (shape: [...])
    """
    # Create support bins
    support = torch.linspace(value_min, value_max, support_size, device=categorical.device)
    
    # Compute expectation
    return (categorical * support).sum(dim=-1)


def cross_entropy_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """Cross-entropy loss for categorical representations.
    
    Args:
        predictions: Predicted logits (shape: [..., support_size])
        targets: Target categorical distributions (shape: [..., support_size])
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Cross-entropy loss
    """
    # Ensure targets sum to 1 (they should already, but just in case)
    targets = targets / (targets.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute log probabilities
    log_probs = F.log_softmax(predictions, dim=-1)
    
    # Cross-entropy: -sum(targets * log(predictions))
    loss = -(targets * log_probs).sum(dim=-1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def categorical_l2_loss(
    predictions: Tensor,
    targets: Tensor,
    support_size: int = 601,
    value_min: float = -300.0,
    value_max: float = 300.0,
    reduction: str = 'mean'
) -> Tensor:
    """L2 loss between scalar values of categorical distributions.
    
    This can be useful for debugging or as an auxiliary loss.
    
    Args:
        predictions: Predicted categorical distributions (shape: [..., support_size])
        targets: Target categorical distributions (shape: [..., support_size])
        support_size: Number of bins in the support
        value_min: Minimum value of the support range
        value_max: Maximum value of the support range
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        L2 loss between scalar values
    """
    pred_scalars = support_to_scalar(predictions, support_size, value_min, value_max)
    target_scalars = support_to_scalar(targets, support_size, value_min, value_max)
    
    loss = (pred_scalars - target_scalars) ** 2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss