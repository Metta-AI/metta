"""
AGaLiTe Improved - Experimental version combining best ideas from optimized/turbo.
This version aims to be more faithful to the AGaLiTe paper while maintaining stability.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.agalite import AGaLiTePolicy
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class AGaLiTeImproved(PyTorchAgentMixin, TransformerWrapper):
    """
    AGaLiTe Improved - Optimized version with enhanced stability.

    This variant provides:
    - Fast mode with stable parameters (eta=2, r=4)
    - Small dropout (0.05) for better generalization
    - Efficient FastAGaLiTeLayer implementation
    - Same architecture as base AGaLiTe but with dropout
    - Option to experiment with token-native processing

    Goal: Provide a stable, production-ready AGaLiTe with minor improvements.
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 2,  # Same as normal AGaLiTe for stability
        eta: int = 2,  # Fast mode value (required for FastAGaLiTeLayer)
        r: int = 4,  # Fast mode value (required for FastAGaLiTeLayer)
        reset_on_terminate: bool = True,
        dropout: float = 0.05,  # Small dropout for generalization
        use_token_native: bool = False,  # Option to use token-native processing
        **kwargs,
    ):
        """Initialize AGaLiTe Improved with stable parameters.

        Args:
            env: Environment
            d_model: Model dimension (256)
            d_head: Head dimension (64)
            d_ffc: Feedforward dimension (1024)
            n_heads: Number of attention heads (4)
            n_layers: Number of transformer layers (2)
            eta: AGaLiTe eta parameter for feature expansion (2)
            r: AGaLiTe r parameter for oscillatory components (4)
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
            # Use regular AGaLiTePolicy with fast mode
            policy = AGaLiTePolicy(
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
