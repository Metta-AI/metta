import logging
from types import SimpleNamespace

import torch
from omegaconf import DictConfig
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

    # Wrap in PytorchAgent to provide the expected interface
    return PytorchAgent(policy)


class PytorchAgent(nn.Module):
    """Adapter to make PyTorch-based policies compatible with MettaAgent interface.

    This wrapper translates between the PyTorch agent interface (which expects
    observations and returns actions/values/logits) and the MettaAgent interface
    (which uses TensorDicts and expects specific key names).

    Key differences handled:
    - Batch dimension management (flattening/reshaping for BPTT)
    - State management (LSTM hidden states)
    - Output key naming (values vs value, etc.)
    - Computing full_log_probs from logits

    Limitation: PyTorch agents don't natively support evaluating given actions,
    so in training mode we use the sampled action probabilities rather than
    evaluating the specific given actions.
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

    def forward(self, td, action=None):
        """MettaAgent-compatible forward method.

        Args:
            td: TensorDict containing at least "env_obs"
            action: Optional action tensor for training mode

        Returns:
            TensorDict with modified outputs matching MettaAgent interface
        """
        import torch
        import torch.nn.functional as F

        # Handle batch dimensions similar to MettaAgent
        if td.batch_dims > 1:
            B, TT = td.batch_size[0], td.batch_size[1]
            # Flatten batch and time dimensions for processing
            td = td.reshape(B * TT)
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size.numel()
            TT = 1
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))

        # Get observations
        obs = td.get("env_obs")

        # Create state dict for PyTorch agent compatibility
        state = {"lstm_h": td.get("lstm_h", None), "lstm_c": td.get("lstm_c", None), "hidden": td.get("hidden", None)}

        # Call the underlying PyTorch agent's forward method
        # Returns: (actions, logprob, entropy, value, logits_list)
        # Note: We always pass action=None because PyTorch agents sample their own actions
        # and don't support evaluating specific given actions
        actions_out, logprob, entropy, value, logits_list = self.policy(obs, state, action=None)

        # Process value tensor
        if value is not None and value.shape[-1] == 1:
            value = value.squeeze(-1)

        # Compute full_log_probs from the first action head's logits
        # This matches how MettaAgent computes it
        if logits_list and isinstance(logits_list, list) and len(logits_list) > 0:
            first_logits = logits_list[0]
            full_log_probs = F.log_softmax(first_logits, dim=-1)
        else:
            # Fallback: create dummy full_log_probs if needed
            full_log_probs = torch.zeros(B * TT, 7, device=td.device)  # 7 is default action space size

        # Store outputs in TensorDict
        if action is None:
            # Inference mode - sample new actions
            td["actions"] = actions_out
            td["act_log_prob"] = logprob
            td["values"] = value  # Note: 'values' in inference mode
            td["full_log_probs"] = full_log_probs
            output_td = td
        else:
            # Training mode - evaluate given actions
            # Since PyTorch agents sample actions, we'll use the sampled values
            # This is a limitation but necessary for compatibility
            td["act_log_prob"] = logprob
            td["entropy"] = entropy
            td["value"] = value  # Note: 'value' in training mode
            td["full_log_probs"] = full_log_probs
            # Reshape back to (B, TT)
            output_td = td.reshape(B, TT)

        # Store updated LSTM states if they exist
        if "lstm_h" in state and state["lstm_h"] is not None:
            output_td["lstm_h"] = state["lstm_h"]
        if "lstm_c" in state and state["lstm_c"] is not None:
            output_td["lstm_c"] = state["lstm_c"]

        return output_td

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Forward to wrapped policy if it has this method."""
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        self.device = device

    def get_agent_experience_spec(self):
        """Provide experience spec for pytorch agents."""
        import torch
        from torchrl.data import Composite, UnboundedDiscrete

        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )

    def reset_memory(self):
        """Reset LSTM memory if present."""
        if hasattr(self.policy, "lstm"):
            # Reset LSTM hidden states if needed
            pass  # PyTorch agents manage their own LSTM state

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
