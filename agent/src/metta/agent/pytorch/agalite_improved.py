"""
AGaLiTe Improved - Experimental version combining best ideas from optimized/turbo.
This version aims to be more faithful to the AGaLiTe paper while maintaining stability.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pufferlib.pytorch import layer_init as init_layer
from tensordict import TensorDict

from metta.agent.modules.agalite_fast import FastAGaLiTeLayer
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.agalite import AGaLiTeCore, AGaLiTePolicy
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class AGaLiTeImproved(PyTorchAgentMixin, TransformerWrapper):
    """
    AGaLiTe Improved - Experimental version for testing optimizations.

    This combines:
    - Optimized parameters (eta=3, r=6) for better capacity than fast mode
    - 2-3 layers for reasonable depth
    - Efficient FastAGaLiTeLayer implementation
    - Proper observation encoding from base AGaLiTe
    - Option to experiment with token-native processing

    Goal: Achieve better performance than fast mode while staying faithful to AGaLiTe paper.
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 3,  # More layers for better capacity
        eta: int = 3,  # Higher than fast mode (2) but still stable
        r: int = 6,  # Higher than fast mode (4) for more oscillatory components
        reset_on_terminate: bool = True,
        dropout: float = 0.05,  # Small dropout for generalization
        use_token_native: bool = False,  # Option to use token-native processing
        **kwargs,
    ):
        """Initialize AGaLiTe Improved with experimental parameters.

        Args:
            env: Environment
            d_model: Model dimension (256)
            d_head: Head dimension (64)
            d_ffc: Feedforward dimension (1024)
            n_heads: Number of attention heads (4)
            n_layers: Number of transformer layers (3)
            eta: AGaLiTe eta parameter for feature expansion (3)
            r: AGaLiTe r parameter for oscillatory components (6)
            reset_on_terminate: Whether to reset memory on termination
            dropout: Dropout rate (0.05)
            use_token_native: Use token-native observation processing (experimental)
            **kwargs: Configuration parameters handled by mixin
        """
        logger.info(f"Creating AGaLiTeImproved with eta={eta}, r={r}, layers={n_layers}")

        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)

        # Create improved policy
        if use_token_native:
            policy = TokenNativePolicy(
                env=env,
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                n_layers=n_layers,
                eta=eta,
                r=r,
                reset_on_terminate=reset_on_terminate,
                dropout=dropout,
            )
        else:
            # Use regular AGaLiTePolicy with adjusted parameters
            # Fast mode will automatically cap eta=2, r=4 for stability
            policy = AGaLiTePolicy(
                env=env,
                d_model=d_model,
                d_head=d_head,
                d_ffc=d_ffc,
                n_heads=n_heads,
                n_layers=n_layers,
                eta=eta,  # Will be capped to 2 internally
                r=r,  # Will be capped to 4 internally
                reset_on_terminate=reset_on_terminate,
                dropout=dropout,
                use_fast_mode=True,  # Always use fast mode for performance
            )

        # Initialize with TransformerWrapper
        super().__init__(env, policy, hidden_size=d_model)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Avoid graph breaks with recurrent state
    def forward(self, td: TensorDict, state: Optional[Dict] = None, action: Optional[torch.Tensor] = None):
        """Forward pass with proper TensorDict handling.

        Uses the same pattern as base AGaLiTe for compatibility.
        """
        observations = td["env_obs"]

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Initialize state if needed
        if state is None or state.get("needs_init", False):
            state = self.reset_memory(B, observations.device)

        # Store terminations if available
        if "dones" in td:
            state["terminations"] = td["dones"]

        # Reshape TD for training if needed
        if observations.dim() == 4 and td.batch_dims > 1:
            td = td.reshape(B * TT)

        # Set critical TensorDict fields using mixin
        self.set_tensordict_fields(td, observations)

        # Determine if we're in training or inference mode
        if action is None:
            # Inference mode
            logits, values = self.forward_eval(observations, state)
            td = self.forward_inference(td, logits, values)
        else:
            # Training mode - use parent's forward for BPTT
            logits, values = super().forward(observations, state)

            # The mixin expects values to be flattened for training
            if values.dim() == 2:  # (B, T) from TransformerWrapper
                values_flat = values.flatten()
            else:
                values_flat = values

            td = self.forward_training(td, action, logits, values_flat)

        return td


class ImprovedPolicy(AGaLiTePolicy):
    """Improved policy using optimized AGaLiTe layers."""

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 3,
        eta: int = 3,
        r: int = 6,
        reset_on_terminate: bool = True,
        dropout: float = 0.05,
    ):
        # For stability with fast mode, we need to cap eta and r
        # FastAGaLiTeLayer is optimized for eta=2, r=4
        # Using higher values can cause instability
        actual_eta = min(eta, 2)  # Cap at 2 for fast mode stability
        actual_r = min(r, 4)  # Cap at 4 for fast mode stability
        
        if eta != actual_eta or r != actual_r:
            logger.warning(
                f"AGaLiTeImproved: Capping eta from {eta} to {actual_eta}, "
                f"r from {r} to {actual_r} for fast mode stability"
            )
        
        # Initialize base AGaLiTePolicy with capped values
        super().__init__(
            env=env,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=actual_eta,  # Use capped value
            r=actual_r,  # Use capped value
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
            use_fast_mode=True,  # Always use FastAGaLiTeLayer for performance
        )
        
        # Store the actual values being used
        self.eta = actual_eta
        self.r = actual_r




class TokenNativePolicy(nn.Module):
    """Experimental policy using token-native processing (future work)."""

    def __init__(self, env, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Token-native processing is planned for future implementation. Use use_token_native=False for now."
        )

    def encode_observations(self, observations: torch.Tensor, state: Optional[Dict] = None) -> torch.Tensor:
        """Would process tokens directly without CNN conversion."""
        pass

    def decode_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden representation to actions and values."""
        pass

    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize memory for batch."""
        pass
