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

import importlib
import logging
from types import SimpleNamespace
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.agent.policy_state import PolicyState
from metta.agent.util.distribution_utils import sample_actions

logger = logging.getLogger("pytorch_adapter")


def create_external_policy(target_str: str, env, hidden_size: int, pytorch_cfg: DictConfig = None):
    """Helper function to create external policies with proper initialization.

    This handles the case where external policies like torch.Recurrent require
    both a Policy and Recurrent wrapper to be created together.

    Args:
        target_str: The target string (e.g., 'metta.agent.external.torch.Recurrent')
        env: Environment namespace with action/observation space info
        hidden_size: Hidden size for the policy
        pytorch_cfg: Full configuration dict for the policy

    Returns:
        Initialized policy ready to load weights
    """
    # Parse the module and class name
    module_path, class_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)

    # Check if this module has both Policy and Recurrent classes
    if hasattr(module, "Policy") and hasattr(module, "Recurrent") and class_name == "Recurrent":
        # This is a PufferLib-style policy that needs both classes initialized
        Policy = module.Policy
        Recurrent = module.Recurrent

        # Extract relevant kwargs from config
        kwargs = {}
        if pytorch_cfg:
            # Get all config params except _target_
            for key, value in pytorch_cfg.items():
                if key != "_target_" and key != "env" and key != "policy":
                    kwargs[key] = value

        # Ensure hidden_size is set
        if "hidden_size" not in kwargs:
            kwargs["hidden_size"] = hidden_size

        # Create the base policy first
        base_policy = Policy(env, **kwargs)

        # Create the recurrent wrapper with the base policy
        input_size = kwargs.get("input_size", kwargs["hidden_size"])
        policy = Recurrent(env, policy=base_policy, input_size=input_size, hidden_size=kwargs["hidden_size"])

        return policy
    else:
        # For other policies, use standard instantiation
        if pytorch_cfg:
            return instantiate(pytorch_cfg, env=env)
        else:
            # Fallback to direct class instantiation
            PolicyClass = getattr(module, class_name)
            return PolicyClass(env, hidden_size=hidden_size)


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: DictConfig = None):
    """Load a PyTorch policy from checkpoint and wrap it in PytorchAdapter.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy (external policy class config)

    Returns:
        PytorchAdapter wrapping the loaded policy
    """
    try:
        weights = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {path}: {e}") from None

    # Validate checkpoint structure
    required_keys = ["policy.actor.0.weight", "policy.network.0.weight"]
    missing_keys = [k for k in required_keys if k not in weights]
    if missing_keys and not ("lstm.weight_ih_l0" in weights or "cell.weight_ih" in weights):
        logger.warning(f"Checkpoint may be incompatible. Missing expected keys: {missing_keys}")

    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        # TODO -- fix all magic numbers
        num_actions, num_action_args = 9, 10
        hidden_size, obs_channels = 512, 22

    # Create environment namespace
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(
            shape=tuple(torch.tensor([obs_channels, 11, 11], dtype=torch.long).tolist())
        ),
    )

    # Instantiate the external policy if config provided, otherwise auto-detect
    if pytorch_cfg and hasattr(pytorch_cfg, "_target_"):
        # Use helper to create policy with proper initialization
        policy = create_external_policy(pytorch_cfg._target_, env, hidden_size, pytorch_cfg)
    else:
        # Auto-detect the model type from checkpoint keys
        if "lstm.weight_ih_l0" in weights or "cell.weight_ih" in weights:
            # This is a Recurrent (LSTMWrapper) model
            from metta.agent.external.torch import Policy, Recurrent

            base_policy = Policy(env, hidden_size=hidden_size)
            policy = Recurrent(env, policy=base_policy, input_size=hidden_size, hidden_size=hidden_size)
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

        # Get max_vec for normalization from the external policy
        self.max_vec = None
        if hasattr(policy, "max_vec"):
            self.max_vec = policy.max_vec
        elif hasattr(policy, "policy") and hasattr(policy.policy, "max_vec"):
            self.max_vec = policy.policy.max_vec
        else:
            # Default max_vec based on external torch.py values
            self.max_vec = torch.tensor(
                [
                    9.0,
                    1.0,
                    1.0,
                    10.0,
                    3.0,
                    254.0,
                    1.0,
                    1.0,
                    235.0,
                    8.0,
                    9.0,
                    250.0,
                    29.0,
                    1.0,
                    1.0,
                    8.0,
                    1.0,
                    1.0,
                    6.0,
                    3.0,
                    1.0,
                    2.0,
                ],
                dtype=torch.float32,
            )

        # Ensure max_vec is 1D for easier indexing
        if self.max_vec is not None and self.max_vec.dim() > 1:
            self.max_vec = self.max_vec.squeeze()

    def forward(
        self, obs: torch.Tensor, state: Optional[PolicyState] = None, action: Optional[torch.Tensor] = None
    ) -> tuple:
        """Forward pass through the wrapped policy.

        Args:
            obs: Observation tensor
            state: Optional LSTM state
            action: Optional action tensor (for training mode)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Sampled action, shape (BT, 2)
            - action_log_prob: Log probability of the sampled action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT, 1)
            - log_probs: Log-softmax of logits, shape (BT, A)
        """
        # Critical fix: Zero out 255 values in observations before processing
        # This matches the preprocessing done in the original PufferLib policy
        obs = obs.clone()  # Clone to avoid modifying the original
        obs[obs == 255] = 0

        # Apply normalization if we have tokenized observations and max_vec
        if self.max_vec is not None and len(obs.shape) == 3 and obs.shape[-1] == 3:
            # Token observations: [B, M, 3] where 3 = (coord, feature_id, feature_value)
            # We need to normalize the feature values based on their feature IDs
            batch_size, num_tokens, _ = obs.shape
            coords = obs[:, :, 0]
            feature_ids = obs[:, :, 1].long()
            feature_values = obs[:, :, 2]

            # Create normalization tensor on the same device
            if self.max_vec.device != obs.device:
                self.max_vec = self.max_vec.to(obs.device)

            # Get normalization values for each feature ID
            # Use a mask to only normalize valid tokens (coord != 0)
            valid_mask = coords != 0

            # Clamp feature IDs to valid range and flatten for indexing
            feature_ids_flat = torch.clamp(feature_ids.view(-1), 0, len(self.max_vec) - 1)

            # Get normalization values for each token
            normalizers_flat = self.max_vec[feature_ids_flat]
            normalizers = normalizers_flat.view(batch_size, num_tokens)

            # Apply normalization only to valid tokens
            normalized_values = torch.where(
                valid_mask & (normalizers > 0), feature_values / normalizers, feature_values
            )

            # Update the observation tensor
            obs[:, :, 2] = normalized_values

        # Convert state format if needed
        if self.is_lstm_wrapper and state is not None:
            # Convert PolicyState to dict format expected by PufferLib
            state_dict = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}
        else:
            state_dict = None

        # Forward through the wrapped policy
        if action is not None:
            # Training mode - use forward_train if available
            if hasattr(self.policy, "forward_train"):
                outputs = self.policy.forward_train(obs, state_dict, action)
            else:
                outputs = self.policy(obs, state_dict)
        else:
            # Inference mode - use forward_eval if available
            if hasattr(self.policy, "forward_eval"):
                outputs = self.policy.forward_eval(obs, state_dict)
            else:
                outputs = self.policy(obs, state_dict)

        # Unpack outputs
        if isinstance(outputs, tuple) and len(outputs) == 2:
            (logits, value), new_state = outputs
        else:
            # Handle case where policy doesn't return state
            logits, value = outputs
            new_state = state_dict

        # Convert state format back if needed
        if self.is_lstm_wrapper and new_state is not None:
            # Convert from dict back to PolicyState
            if isinstance(new_state, dict):
                if state is not None:
                    state.lstm_h = new_state["lstm_h"]
                    state.lstm_c = new_state["lstm_c"]
                else:
                    # Create new state if none was provided
                    state = PolicyState(lstm_h=new_state["lstm_h"], lstm_c=new_state["lstm_c"])
            else:
                # new_state might already be a PolicyState or tensor
                if state is None:
                    state = PolicyState()
        elif state is None:
            # Ensure we have a state object
            state = PolicyState()

        # Convert logits to MettaAgent-compatible format
        if isinstance(logits, list):
            # Multi-discrete actions - concatenate logits
            logits = torch.cat(logits, dim=-1)

        # Ensure value has correct shape
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        # Sample actions and compute probabilities
        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)

        # Convert logit index to (action_type, action_param) format
        # This is necessary for MettaAgent compatibility
        action = self._convert_logit_index_to_action(action_logit_index)

        return action, action_log_prob, entropy, value, log_probs

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
        action_indices, logprob, logits_entropy = sample_logits(hidden, action)

        # Convert action indices to (action_type, action_param) format
        # For standard policies, we assume single discrete actions
        if action_indices.dim() == 1:
            # Create action pairs (action_type=indices, action_param=0)
            actions = torch.stack([action_indices, torch.zeros_like(action_indices)], dim=-1)
        else:
            actions = action_indices

        return actions, logprob, logits_entropy, critic, hidden

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
            logits = torch.cat([log for log in logits], dim=-1)

            # Sample actions and compute log probs
        action_indices, logprob, entropy = sample_logits(logits, action)

        # Convert action indices to (action_type, action_param) format
        # This mimics MettaAgent's _convert_logit_index_to_action
        if hasattr(self.policy, "policy") and hasattr(self.policy.policy, "actor"):
            # Get action space info from the policy
            action_nvec = [head.out_features for head in self.policy.policy.actor]

            # Build action index tensor (precompute for efficiency)
            action_index = []
            for action_type_idx, max_param in enumerate(action_nvec):
                for j in range(max_param):
                    action_index.append([action_type_idx, j])

            action_index_tensor = torch.tensor(action_index, device=action_indices.device, dtype=torch.int32)

            # Convert indices to action pairs
            actions = action_index_tensor[action_indices]
        else:
            # Fallback: assume actions are already in the right format or single discrete
            # For single discrete, we need to expand to (batch, 2) format
            if action_indices.dim() == 1:
                # Create dummy action pairs (action_type=indices, action_param=0)
                actions = torch.stack([action_indices, torch.zeros_like(action_indices)], dim=-1)
            else:
                actions = action_indices

        # Ensure value has correct shape
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        return actions, logprob, entropy, value, logits

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
                action_indices, logprob, entropy = sample_logits(logits, action)

                # Convert to (action_type, action_param) format
                if action_indices.dim() == 1:
                    actions = torch.stack([action_indices, torch.zeros_like(action_indices)], dim=-1)
                else:
                    actions = action_indices

                return actions, logprob, entropy, value, logits
            else:
                # Might already be in the right format
                return result
        else:
            return (
                self.policy.forward_train(x, state) if hasattr(self.policy, "forward_train") else self.policy(x, state)
            )

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Forward to wrapped policy if it has this method, and set up action conversion."""
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        self.device = device

        # Set up action conversion similar to MettaAgent
        self.action_max_params = action_max_params
        self.action_names = action_names

        # Precompute cumulative sums for faster conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=device, dtype=torch.long), dim=0
        )

        # Create action_index tensor for converting logit indices to action pairs
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=device, dtype=torch.int32)

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs using tensor indexing.

        Args:
            action_logit_index: Tensor of shape [B*T] containing flattened action indices

        Returns:
            action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs
        """
        if not hasattr(self, "action_index_tensor"):
            # Fallback for policies that haven't called activate_actions
            # Assume default action space [9, 10] for MettaGrid
            if not hasattr(self, "_default_action_index"):
                action_index = []
                for i in range(9):  # 9 action types
                    for j in range(11):  # up to 10 params (0-10)
                        action_index.append([i, j])
                self._default_action_index = torch.tensor(
                    action_index, device=action_logit_index.device, dtype=torch.int32
                )
            return self._default_action_index[action_logit_index]

        return self.action_index_tensor[action_logit_index]

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
