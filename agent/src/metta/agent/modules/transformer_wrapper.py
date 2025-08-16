"""
Transformer wrapper for recurrent policies in PufferLib/Metta infrastructure.
Similar to LSTMWrapper but handles transformer-style memory states and full sequences.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransformerWrapper(nn.Module):
    """
    Wrapper for transformer-based recurrent policies.

    Key differences from LSTM:
    - Processes entire BPTT sequences as context (not step-by-step)
    - Maintains complex memory state (not just h/c)
    - Handles termination-aware memory resets
    """

    def __setstate__(self, state):
        """Ensure transformer memory states are properly initialized after loading from checkpoint.
        
        This prevents hangs and batch size mismatches when resuming from checkpoints.
        Similar to LSTM's __setstate__, we clear memory to handle batch size changes.
        """
        self.__dict__.update(state)
        # Reset memory states when loading from checkpoint to avoid batch size mismatch
        if not hasattr(self, "transformer_memory"):
            self.transformer_memory = {}
        # Clear any existing states to handle batch size mismatches
        self.transformer_memory.clear()

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
        
        # Store memory states for each environment (similar to LSTM)
        self.transformer_memory = {}

    def forward_eval(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference (single timestep).

        For transformers, we still need to handle single-step inference
        but can use the memory from previous sequences.

        Args:
            observations: Observations of shape (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates
        """
        # Encode observations
        hidden = self.policy.encode_observations(observations, state=state)

        # Get transformer memory from per-environment storage
        env_id = state.get("env_id", 0)
        memory = self.transformer_memory.get(env_id, None)

        # Process through transformer (single timestep)
        # Add time dimension for transformer
        hidden = hidden.unsqueeze(0)  # (1, B, hidden_size)

        # Check for terminations in state
        terminations = state.get("terminations", torch.zeros(1, observations.shape[0], device=hidden.device))
        if terminations.dim() == 1:
            terminations = terminations.unsqueeze(0)

        # Forward through transformer
        hidden, new_memory = self.policy.transformer(hidden, terminations, memory)

        # Remove time dimension
        hidden = hidden.squeeze(0)

        # Store updated memory in per-environment storage
        self.transformer_memory[env_id] = new_memory
        state["transformer_memory"] = new_memory
        state["hidden"] = hidden

        # Decode actions
        logits, values = self.policy.decode_actions(hidden)

        return logits, values

    def forward(self, observations: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (handles full BPTT sequences).

        Key difference from LSTM: Transformers process the entire sequence
        at once as context, not step-by-step.

        Args:
            observations: Observations of shape (B, T, ...) or (B, ...)
            state: Dictionary containing transformer memory state

        Returns:
            logits: Action logits
            values: Value estimates of shape (B, T) or (B,)
        """
        x = observations
        
        # Get environment ID for memory tracking
        training_env_id_start = state.get("training_env_id_start", 0)
        if isinstance(training_env_id_start, torch.Tensor):
            training_env_id_start = training_env_id_start.item()
        
        # Get memory from per-environment storage
        memory = self.transformer_memory.get(training_env_id_start, None)

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

        # Forward through transformer with full sequence as context
        hidden, new_memory = self.policy.transformer(hidden, terminations, memory)

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

        # Update state and per-environment storage with detached memory
        # Critical: detach memory to prevent infinite gradient accumulation
        if new_memory is not None:
            detached_memory = self._detach_memory(new_memory)
            self.transformer_memory[training_env_id_start] = detached_memory
            state["transformer_memory"] = new_memory  # Keep attached for current pass
        state["hidden"] = hidden

        return logits, values

    def reset_memory(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Initialize memory for a batch of environments.

        This method can be called in two ways:
        1. With explicit batch_size (and optionally device) for initialization
        2. Without arguments (when MettaAgent calls it during reset)

        When called without arguments, we return empty state that will be
        initialized lazily on first forward pass.

        Args:
            batch_size: Optional number of parallel environments
            device: Optional device to create tensors on (ignored, kept for compatibility)

        Returns:
            Initial memory state dictionary
        """
        if batch_size is None:
            # Called by MettaAgent.reset_memory() without args
            # Return empty state - will be initialized lazily on first forward
            return {
                "transformer_memory": None,
                "hidden": None,
                "needs_init": True,  # Flag to indicate lazy initialization needed
            }

        # Explicit initialization with batch_size
        # Note: We ignore the device parameter and let the policy handle device placement
        # internally, similar to how LSTMWrapper works. The device will be inferred
        # from the tensors being processed (observations.device)
        if hasattr(self.policy, "initialize_memory"):
            memory = self.policy.initialize_memory(batch_size)
        else:
            memory = None

        return {
            "transformer_memory": memory,
            "hidden": None,
            "needs_init": False,
        }
    
    def has_memory(self):
        """Indicate that this policy has memory (transformer states)."""
        return True
    
    def get_memory(self):
        """Get current transformer memory states for checkpointing.
        
        Returns a pickle-safe representation of the memory state.
        """
        # Return the per-environment memory storage
        if self.transformer_memory:
            # Make sure memory is CPU and detached for safe pickling
            safe_memory = {}
            for env_id, mem in self.transformer_memory.items():
                safe_memory[env_id] = self._make_pickle_safe(mem)
            return safe_memory
        return {}
    
    def set_memory(self, memory):
        """Set transformer memory states from checkpoint."""
        # Note: memory will be properly reinitialized by __setstate__ if needed
        if isinstance(memory, dict):
            self.transformer_memory = memory
        else:
            self.transformer_memory = {}
    
    def reset_memory(self):
        """Reset all transformer memory states."""
        self.transformer_memory.clear()
    
    def reset_env_memory(self, env_id):
        """Reset transformer memory for a specific environment."""
        if env_id in self.transformer_memory:
            del self.transformer_memory[env_id]
    
    def _make_pickle_safe(self, obj):
        """Make an object pickle-safe by detaching and moving to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, tuple):
            return tuple(self._make_pickle_safe(item) for item in obj)
        elif isinstance(obj, list):
            return [self._make_pickle_safe(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_pickle_safe(v) for k, v in obj.items()}
        else:
            return obj
    
    def _detach_memory(self, memory):
        """Detach memory tensors to prevent gradient accumulation.
        
        Critical for preventing memory leaks and training instability.
        """
        if memory is None:
            return None
        elif isinstance(memory, torch.Tensor):
            return memory.detach()
        elif isinstance(memory, tuple):
            return tuple(self._detach_memory(item) for item in memory)
        elif isinstance(memory, dict):
            return {k: self._detach_memory(v) for k, v in memory.items()}
        else:
            return memory
