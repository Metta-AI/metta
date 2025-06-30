"""Adapter for loading external PyTorch policies into Metta.

This module provides a unified adapter system that allows external PyTorch policies
(particularly from PufferLib) to be used within Metta's training and evaluation framework.

The main class, PytorchAdapter, automatically detects the type of external policy and
applies appropriate conversions for compatibility with MettaAgent.

Key features:
- Handles PufferLib LSTMWrapper policies without modification (e.g., torch.py)
- Converts between Metta's PolicyState and PufferLib's dict state format
- Works with Metta's native token observations [B, M, 3]
- Provides method forwarding for MettaAgent compatibility
"""

import logging
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.agent.policy_state import PolicyState

logger = logging.getLogger("pytorch_adapter")


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: DictConfig = None):
    """Load a PyTorch policy from checkpoint and wrap it in PytorchAdapter.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy (external policy class config)

    Returns:
        PytorchAdapter wrapping the loaded policy
    """
    weights = torch.load(path, map_location=device, weights_only=True)

    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        # TODO -- fix all magic numbers
        num_actions, num_action_args = 9, 10
        _, obs_channels = 128, 34

    # Create environment namespace
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(
            shape=tuple(torch.tensor([obs_channels, 11, 11], dtype=torch.long).tolist())
        ),
    )

    # Instantiate the external policy if config provided, otherwise create a basic wrapper
    if pytorch_cfg and hasattr(pytorch_cfg, "_target_"):
        # Pass policy=None and let the Recurrent class create the policy
        policy = instantiate(pytorch_cfg, env=env, policy=None)
    else:
        # For backwards compatibility with direct checkpoint loading
        policy = create_basic_policy(env, weights)

    policy.load_state_dict(weights)
    policy = PytorchAdapter(policy).to(device)
    return policy


def create_basic_policy(env, weights):
    """Create a basic policy structure when no external policy class is specified."""

    class BasicPolicy(nn.Module):
        def __init__(self, env):
            super().__init__()
            # This is a placeholder - the actual modules will be loaded from checkpoint
            self.env = env

        def forward(self, obs, state):
            raise NotImplementedError("Basic policy forward not implemented")

    return BasicPolicy(env)


class PytorchAdapter(nn.Module):
    """Unified adapter to make external PyTorch policies compatible with MettaAgent interface.

    This adapter wraps policies from external sources (e.g., PufferLib) and translates
    their outputs to match the expected MettaAgent interface. It handles:
    - Different naming conventions (critic→value, hidden→logits)
    - State management for LSTM policies
    - Method forwarding for MettaAgent compatibility
    - PufferLib LSTMWrapper patterns
    - Token observation handling
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.hidden_size = getattr(policy, "hidden_size", 256)

        # Check if this is a PufferLib LSTMWrapper style policy
        self.is_lstm_wrapper = hasattr(policy, "cell") and hasattr(policy, "policy")

        # For LSTM policies, point to the actual LSTM module
        if self.is_lstm_wrapper:
            self.lstm = getattr(policy, "cell", None)
        else:
            self.lstm = getattr(policy, "lstm", None)

        self.components = nn.ModuleDict()  # Empty for compatibility

    def forward(self, obs: torch.Tensor, state: PolicyState, action=None):
        """Forward pass with MettaAgent-compatible interface.

        Handles both standard PyTorch policies and PufferLib LSTMWrapper policies.
        """
        if self.is_lstm_wrapper:
            return self._forward_lstm_wrapper(obs, state, action)
        else:
            return self._forward_standard(obs, state, action)

    def _forward_standard(self, obs: torch.Tensor, state: PolicyState, action=None):
        """Handle standard PyTorch policies without LSTMWrapper."""
        # Handle different possible forward signatures
        if hasattr(self.policy, "forward") and callable(self.policy.forward):
            # Check the forward signature
            import inspect

            sig = inspect.signature(self.policy.forward)
            params = list(sig.parameters.keys())

            # Call with appropriate arguments based on signature
            if "state" in params:
                result = self.policy(obs, state)
            else:
                result = self.policy(obs)

            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                hidden, critic = result
            else:
                # Assume it returns (actions, value) or similar
                hidden, critic = result[0], result[1] if len(result) > 1 else torch.zeros(obs.shape[0], 1)
        else:
            raise NotImplementedError(f"Policy {type(self.policy)} does not have a callable forward method")

        # Sample actions from logits
        action, logprob, logits_entropy = sample_logits(hidden, action)

        # Convert action indices back to (action_type, action_param) format if needed
        # This is handled by MettaAgent's _convert_logit_index_to_action
        # For now, just return the raw indices
        return action, logprob, logits_entropy, critic, hidden

    def _forward_lstm_wrapper(self, obs: torch.Tensor, state: PolicyState, action=None):
        """Handle PufferLib LSTMWrapper style policies."""
        # Convert Metta PolicyState to LSTMWrapper state format (dict)
        if hasattr(state, "lstm_h") and state.lstm_h is not None:
            h, c = state.lstm_h, state.lstm_c
            # Handle shape differences between training and inference
            if len(h.shape) == 3:  # Training format [1, B, hidden_size]
                h, c = h.squeeze(0), c.squeeze(0)  # Convert to [B, hidden_size]
            lstm_state = {"lstm_h": h, "lstm_c": c}
        else:
            # PufferLib expects a dict with lstm_h and lstm_c keys
            lstm_state = {"lstm_h": None, "lstm_c": None}

        # Determine if we're in training or inference mode
        if action is not None or (obs.dim() > 3 and hasattr(self.policy, "forward")):
            # Training mode - use forward method which handles time dimension
            logits, value = self.policy.forward(obs, lstm_state)
        else:
            # Inference mode - use forward_eval for efficiency
            if hasattr(self.policy, "forward_eval"):
                logits, value = self.policy.forward_eval(obs, lstm_state)
            else:
                logits, value = self.policy.forward(obs, lstm_state)

        # Update state from the dict (LSTMWrapper modifies it in-place)
        state.lstm_h = lstm_state.get("lstm_h")
        state.lstm_c = lstm_state.get("lstm_c")
        state.hidden = lstm_state.get("hidden", state.lstm_h)

        # Convert to MettaAgent format
        if isinstance(logits, list):
            # For multi-discrete actions, concatenate logits
            logits = torch.cat([l for l in logits], dim=-1)

        # Sample actions and compute log probs
        action, logprob, entropy = sample_logits(logits, action)

        # Ensure value has correct shape
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        return action, logprob, entropy, value, logits

    def _forward_train_with_state_conversion(self, x, state, action=None):
        """Helper to handle state conversion for training."""
        if hasattr(state, "lstm_h"):
            # Convert PolicyState to dict for compatibility
            state_dict = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c, "hidden": getattr(state, "hidden", None)}

            # Check if the policy expects the action in forward_train
            import inspect

            if hasattr(self.policy, "forward_train"):
                sig = inspect.signature(self.policy.forward_train)
                if "action" in sig.parameters:
                    result = self.policy.forward_train(x, state_dict, action)
                else:
                    result = self.policy.forward_train(x, state_dict)
            else:
                result = self.policy(x, state_dict)

            # Update original state
            state.lstm_h = state_dict.get("lstm_h")
            state.lstm_c = state_dict.get("lstm_c")
            state.hidden = state_dict.get("hidden")

            # Handle different return formats from forward_train
            if isinstance(result, tuple) and len(result) == 2:
                # Standard (logits, value) format
                logits, value = result
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=-1)
                action, logprob, entropy = sample_logits(logits, action)
                return action, logprob, entropy, value, logits
            else:
                # Might already be in the right format
                return result
        else:
            return (
                self.policy.forward_train(x, state) if hasattr(self.policy, "forward_train") else self.policy(x, state)
            )

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Forward to wrapped policy if it has this method."""
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        self.device = device

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss."""
        if hasattr(self.policy, "l2_reg_loss"):
            return self.policy.l2_reg_loss()
        return torch.tensor(0.0, device=getattr(self, "device", "cpu"), dtype=torch.float32)

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss."""
        if hasattr(self.policy, "l2_init_loss"):
            return self.policy.l2_init_loss()
        return torch.tensor(0.0, device=getattr(self, "device", "cpu"), dtype=torch.float32)

    def update_l2_init_weight_copy(self):
        """Update L2 initialization weight copy."""
        if hasattr(self.policy, "update_l2_init_weight_copy"):
            self.policy.update_l2_init_weight_copy()

    def clip_weights(self):
        """Clip weights."""
        if hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []


# Keep ExternalPolicyAdapter as an alias for backwards compatibility
ExternalPolicyAdapter = PytorchAdapter
