import logging
from types import SimpleNamespace

import torch
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn

from metta.agent.policy_state import PolicyState
from metta.common.util.instantiate import instantiate

logger = logging.getLogger("policy")


def load_pytorch_policy(path: str, device: str = "cpu", pytorch_cfg: DictConfig = None):
    """Load a PyTorch policy from checkpoint and wrap it in PytorchAgent.

    Args:
        path: Path to the checkpoint file
        device: Device to load the policy on
        pytorch_cfg: Configuration for the PyTorch policy with _target_ field

    Returns:
        PytorchAgent wrapping the loaded policy
    """
    weights = torch.load(path, map_location=device, weights_only=True)

    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        logger.warning("Using defaults from config")
        num_actions = 9
        hidden_size = 512
        num_action_args = 10
        obs_channels = 24

    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=(11, 11, obs_channels)),
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
    )

    # Use common instantiate function
    if pytorch_cfg is None:
        # Default to Recurrent policy if no config provided
        from metta.agent.external.example import Recurrent

        policy = Recurrent(
            env=env,
            policy=None,
            hidden_size=hidden_size,
            conv_depth=2,
            conv_channels=32,
        )
    else:
        # Use the common instantiate utility
        policy = instantiate(pytorch_cfg, env=env, policy=None)

    policy.load_state_dict(weights)

    # Wrap in PytorchAgent and move to device
    policy = PytorchAgent(policy).to(device)
    return policy


class PytorchAgent(nn.Module):
    """Adapter to make torch.nn.Module-based policies compatible with MettaAgent interface.

    This adapter wraps policies loaded from checkpoints and translates their
    outputs to match the expected MettaAgent interface, handling naming
    differences like critic→value, hidden→logits, etc.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        self.hidden_size = getattr(policy, "hidden_size", 256)
        self.lstm = getattr(policy, "lstm", None)  # Point to the actual LSTM module if it exists
        self.components = nn.ModuleDict()  # Empty for compatibility

    def forward(self, obs: torch.Tensor, state: PolicyState, action=None):
        """Uses variable names from LSTMWrapper. Translating for Metta:
        critic -> value
        logprob -> logprob_act
        hidden -> logits then, after sample_logits(), log_sftmx_logits
        """
        hidden, critic = self.policy(obs, state)  # using variable names from LSTMWrapper
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
        """Initialize to environment - forward to wrapped policy if it has this method."""
        # is_training parameter is deprecated and ignored - mode is auto-detected

        # TODO: This hasattr pattern is a transitional state to support both old and new interfaces.
        # Once all policies have been migrated to implement initialize_to_environment,
        # we should remove these checks and make the interface mandatory.
        if hasattr(self.policy, "initialize_to_environment"):
            self.policy.initialize_to_environment(features, action_names, action_max_params, device)
        elif hasattr(self.policy, "activate_actions"):
            # Fallback to old interface if available
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
