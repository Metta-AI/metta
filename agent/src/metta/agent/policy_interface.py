"""
PolicyInterface: Abstract base class defining the required interface for all policies.

This ensures that all policies (ComponentPolicy, PyTorch agents with mixin, etc.)
implement the required methods that MettaAgent depends on.
"""

from abc import ABC, abstractmethod
from typing import Optional, list, dict

import torch
import torch.nn as nn
from tensordict import TensorDict


class PolicyInterface(ABC, nn.Module):
    """
    Abstract base class defining the interface that all policies must implement.
    
    This ensures compatibility with MettaAgent and the training pipeline.
    All policies (ComponentPolicy, PyTorch agents) should inherit from or
    implement this interface.
    """
    
    # These attributes will be set by MettaAgent.activate_actions()
    action_index_tensor: Optional[torch.Tensor] = None
    cum_action_max_params: Optional[torch.Tensor] = None
    
    @abstractmethod
    def forward(self, td: TensorDict, state=None, action=None) -> TensorDict:
        """
        Forward pass of the policy.
        
        Args:
            td: TensorDict containing observations and other data
            state: Optional state for recurrent policies
            action: Optional action for training mode
            
        Returns:
            TensorDict with actions, values, and other outputs
        """
        pass
    
    @abstractmethod
    def clip_weights(self):
        """
        Clip weights to prevent large updates during training.
        
        Called by the trainer after optimizer steps when clip_range > 0.
        """
        pass
    
    @abstractmethod
    def l2_init_loss(self) -> torch.Tensor:
        """
        Calculate L2 initialization loss for regularization.
        
        Returns:
            L2 loss between current weights and initial weights
        """
        pass
    
    @abstractmethod
    def update_l2_init_weight_copy(self):
        """Update the stored initial weights for L2 regularization."""
        pass
    
    @abstractmethod
    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """
        Compute weight metrics for monitoring.
        
        Args:
            delta: Small value for numerical stability
            
        Returns:
            List of metric dictionaries
        """
        pass
    
    @abstractmethod
    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete indices.
        
        Args:
            flattened_action: Actions as (action_type, action_param) pairs
            
        Returns:
            Indices into flattened action space
        """
        pass
    
    @abstractmethod
    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs.
        
        Args:
            action_logit_index: Indices into flattened action space
            
        Returns:
            Action tensor with (action_type, action_param) pairs
        """
        pass
    
    def activate_action_embeddings(self, full_action_names: list[str], device):
        """
        Activate action embeddings if the policy uses them.
        
        Args:
            full_action_names: List of action names
            device: Device for tensors
        """
        # Default implementation - override if policy has action embeddings
        pass
    
    def reset_memory(self):
        """Reset memory for recurrent policies."""
        # Default implementation - override for recurrent policies
        pass
    
    def get_memory(self) -> dict:
        """Get memory state for recurrent policies."""
        # Default implementation - override for recurrent policies
        return {}
    
    def has_memory(self) -> bool:
        """Check if policy has memory/recurrent state."""
        # Default implementation - override for recurrent policies
        return False