import logging
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.common.util.instantiate import instantiate

logger = logging.getLogger("policy")


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: DictConfig | None = None) -> "PytorchAgent":
    """Load a PyTorch policy from checkpoint and wrap it in PytorchAgent.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy with _target_ field

    Returns:
        PytorchAgent wrapping the loaded policy
    """

    try:
        weights = torch.load(path, map_location=device, weights_only=True)

        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        logger.warning("Using defaults from config")
        num_actions = 9
        hidden_size = 512
        num_action_args = 10
        obs_channels = 22  # Updated default to 22 channels

    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
    )

    # Use common instantiate function
    if pytorch_cfg is None:
        # Default to Recurrent policy if no config provided
        from metta.agent.external.example import Policy, Recurrent

        # Create the Policy first
        policy = Policy(
            env=env,
            cnn_channels=128,
            hidden_size=hidden_size,
        )

        # Then wrap it in Recurrent
        policy = Recurrent(
            env=env,
            policy=policy,
            input_size=512,
            hidden_size=hidden_size,
        )
    else:
        # Use the common instantiate utility
        policy = instantiate(pytorch_cfg, env=env, policy=None)
    try:
        policy.load_state_dict(weights)
    except Exception as e:
        logger.warning(f"Failed automatic load from weights: {e}")
        logger.warning("Using randomly initialized weights")

    return policy


class PytorchAgent(nn.Module):
    """Adapter to make torch.nn.Module-based policies compatible with MettaAgent interface.

    This adapter wraps policies loaded from checkpoints and translates their
    outputs to match the expected MettaAgent interface, handling naming
    differences like critic→value, hidden→logits, etc.
    """

    # Default feature normalizations for legacy policies using max_vec
    DEFAULT_FEATURE_NORMALIZATIONS = {
        "type_id": 9.0,
        "agent:group": 1.0,
        "hp": 1.0,
        "agent:frozen": 10.0,
        "agent:orientation": 3.0,
        "agent:color": 254.0,
        "converting": 1.0,
        "swappable": 1.0,
        "episode_completion_pct": 235.0,
        "last_action": 8.0,
        "last_action_arg": 9.0,
        "last_reward": 250.0,
        "agent:glyph": 29.0,
        "resource_rewards": 1.0,
        # Inventory features (positions 14-21)
        "inv:0": 1.0,
        "inv:1": 8.0,
        "inv:2": 1.0,
        "inv:3": 1.0,
        "inv:4": 6.0,
        "inv:5": 3.0,
        "inv:6": 1.0,
        "inv:7": 2.0,
    }

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.hidden_size = getattr(policy, "hidden_size", 256)
        self.lstm = getattr(policy, "lstm", None)  # Point to the actual LSTM module if it exists
        self.components = nn.ModuleDict()  # Empty for compatibility

    def forward(self, obs: torch.Tensor, state: Any, action=None):
        """Uses variable names from LSTMWrapper. Translating for Metta:
        critic -> value
        logprob -> logprob_act
        hidden -> logits then, after sample_logits(), log_sftmx_logits
        """
        # For tokenized observations, we need to handle them differently
        # The pufferlib Recurrent/LSTMWrapper expects already-encoded observations,
        # but we have raw tokenized observations that need to be processed first

        if hasattr(self.policy, "policy") and hasattr(self.policy.policy, "encode_observations"):
            # This is a wrapped policy (e.g., Recurrent wrapping Policy)
            # Encode observations first
            hidden = self.policy.policy.encode_observations(obs, state)

            # Handle LSTM state
            h, c = state.lstm_h, state.lstm_c
            if h is not None:
                if len(h.shape) == 3:
                    h, c = h.squeeze(), c.squeeze()
                assert h.shape[0] == c.shape[0] == obs.shape[0], "LSTM state must be (h, c)"
                lstm_state = (h, c)
            else:
                lstm_state = None

            # LSTM forward pass
            if hasattr(self.policy, "cell"):
                hidden, c = self.policy.cell(hidden, lstm_state)
                # Update state
                state.hidden = hidden
                state.lstm_h = hidden
                state.lstm_c = c

            # Decode actions
            logits, critic = self.policy.policy.decode_actions(hidden)

            # sample_logits expects a list of logits for multi-discrete action spaces
            hidden = logits  # Keep as list for sample_logits

        else:
            # Fallback to original behavior
            state_dict = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c, "hidden": getattr(state, "hidden", None)}

            (hidden, critic), _ = self.policy(obs, state_dict)

            # Update state
            if "lstm_h" in state_dict:
                state.lstm_h = state_dict["lstm_h"]
            if "lstm_c" in state_dict:
                state.lstm_c = state_dict["lstm_c"]
            if "hidden" in state_dict and state_dict["hidden"] is not None:
                state.hidden = state_dict["hidden"]

        # Sample actions from logits
        action, logprob, logits_entropy = sample_logits(hidden, action)
        return action, logprob, logits_entropy, critic, hidden

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Forward to wrapped policy if it has this method."""
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        self.device = device

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """Initialize to environment - handle max_vec normalization for legacy policies."""
        # is_training parameter is deprecated and ignored - mode is auto-detected

        # Handle max_vec normalization for policies that use it
        target_policy = self._get_inner_policy()
        if hasattr(target_policy, "max_vec") and hasattr(target_policy, "num_layers"):
            self._update_max_vec_normalizations(target_policy, features, device)

        # Forward to wrapped policy
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(features, action_names, action_max_params, device)
        self.device = device

    def _get_inner_policy(self):
        """Get the inner policy (unwrap if this is a wrapped policy like Recurrent)."""
        target_policy = self.policy
        if hasattr(self.policy, "policy") and hasattr(self.policy.policy, "max_vec"):
            # This is a wrapped policy, get the inner policy
            target_policy = self.policy.policy
        return target_policy

    def _update_max_vec_normalizations(self, policy, features: dict[str, dict], device):
        """Update max_vec based on feature normalizations for legacy policies."""
        # Don't update if the policy has its own feature normalization system
        if hasattr(policy, "feature_normalizations"):
            return

        # Create max_vec based on current environment's feature IDs
        max_values = [1.0] * policy.num_layers  # Default normalization

        # Map our known features to the environment's feature IDs
        for feature_name, feature_props in features.items():
            if "id" in feature_props and 0 <= feature_props["id"] < policy.num_layers:
                feature_id = feature_props["id"]

                # Check if this is a feature we know about
                if feature_name in self.DEFAULT_FEATURE_NORMALIZATIONS:
                    # Use our empirically determined normalization
                    max_values[feature_id] = self.DEFAULT_FEATURE_NORMALIZATIONS[feature_name]
                elif feature_name.startswith("inv:") and "inv:0" in self.DEFAULT_FEATURE_NORMALIZATIONS:
                    # For unknown inventory items, use a default inventory normalization
                    max_values[feature_id] = 100.0  # DEFAULT_INVENTORY_NORMALIZATION
                elif "normalization" in feature_props:
                    # Use environment's normalization for unknown features
                    max_values[feature_id] = feature_props["normalization"]

        # Update max_vec with the mapped values
        new_max_vec = torch.tensor(max_values, dtype=torch.float32, device=device)[None, :, None, None]
        policy.max_vec.data = new_max_vec

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

    def clip_weights(self):
        """Clip weights."""
        if hasattr(self.policy, "clip_weights"):
            self.policy.clip_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for analysis."""
        if hasattr(self.policy, "compute_weight_metrics"):
            return self.policy.compute_weight_metrics(delta)
        return []
