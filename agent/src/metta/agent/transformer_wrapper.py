"""
Transformer wrapper for recurrent policies in PufferLib/Metta infrastructure.
Similar to LSTMWrapper but handles transformer-style memory states.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class TransformerWrapper(nn.Module):
    """
    Wrapper for transformer-based recurrent policies.

    This wrapper handles:
    - Batch dimension management for BPTT
    - Memory state persistence across timesteps
    - Proper reshaping for vectorized environments
    - Compatible with PufferLib's training infrastructure
    """

    def __init__(self, env, policy, hidden_size: int = 256):
        """
        Initialize the transformer wrapper.

        Args:
            env: The environment (for observation space info)
            policy: The policy network (must have encode_observations and decode_actions)
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.hidden_size = hidden_size
        self.is_continuous = getattr(policy, "is_continuous", False)

        # Initialize weights if needed
        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

    def forward_eval(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference (single timestep).

        Args:
            observations: Observations of shape (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates
        """
        # Encode observations
        hidden = self.policy.encode_observations(observations, state=state)

        # Get transformer memory if it exists
        memory = state.get("transformer_memory", None)

        # Process through transformer (single timestep)
        # Add time dimension for transformer
        hidden = hidden.unsqueeze(0)  # (1, B, hidden_size)

        # Check for terminations in state
        terminations = state.get("terminations", torch.zeros(1, observations.shape[0], device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0)

        # Forward through transformer
        if hasattr(self.policy, "transformer"):
            hidden, new_memory = self.policy.transformer(hidden, terminations, memory)
        else:
            # Fallback for policies without explicit transformer
            new_memory = memory

        # Remove time dimension
        hidden = hidden.squeeze(0)

        # Store updated memory
        state["transformer_memory"] = new_memory
        state["hidden"] = hidden

        # Decode actions
        logits, values = self.policy.decode_actions(hidden)

        return logits, values

    def forward(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (handles sequences with BPTT).

        Args:
            observations: Observations of shape (B, T, ...) or (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates of shape (B, T) or (B,)
        """
        x = observations
        memory = state.get("transformer_memory", None)

        # Determine input shape
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)

        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        # Determine batch and time dimensions
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        # Reshape for encoding: (B*T, ...)
        x = x.reshape(B * TT, *space_shape)

        # Encode observations
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.hidden_size), (
            f"Expected shape ({B * TT}, {self.hidden_size}), got {hidden.shape}"
        )

        # Reshape for transformer: (B*T, hidden) -> (T, B, hidden)
        if TT > 1:
            hidden = hidden.view(B, TT, -1).transpose(0, 1)
        else:
            hidden = hidden.unsqueeze(0)

        # Get terminations if available
        terminations = state.get("terminations", torch.zeros(TT, B, device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0).expand(TT, -1)
        elif terminations.dim() == 2 and terminations.shape[0] == B:
            # (B, T) -> (T, B)
            terminations = terminations.transpose(0, 1)

        # Forward through transformer
        if hasattr(self.policy, "transformer"):
            hidden, new_memory = self.policy.transformer(hidden, terminations, memory)
        else:
            new_memory = memory

        # Convert back to (B, T, hidden) or (B, hidden)
        if TT > 1:
            hidden = hidden.transpose(0, 1)  # (T, B, hidden) -> (B, T, hidden)
            flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        else:
            hidden = hidden.squeeze(0)  # (1, B, hidden) -> (B, hidden)
            flat_hidden = hidden

        # Decode actions
        logits, values = self.policy.decode_actions(flat_hidden)

        # Reshape values
        if TT > 1:
            values = values.reshape(B, TT)

        # Update state
        state["transformer_memory"] = new_memory
        state["hidden"] = hidden

        return logits, values

    def reset_memory(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """
        Initialize memory for a batch of environments.

        Args:
            batch_size: Number of parallel environments
            device: Device to create tensors on

        Returns:
            Initial memory state dictionary
        """
        if hasattr(self.policy, "initialize_memory"):
            memory = self.policy.initialize_memory(batch_size, device)
        else:
            memory = None

        return {
            "transformer_memory": memory,
            "hidden": None,
        }
