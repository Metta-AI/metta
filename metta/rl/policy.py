import logging
from types import SimpleNamespace
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn


def load_policy(path: str, device: str = "cpu", puffer: Optional[DictConfig] = None):
    weights = torch.load(path, map_location=device, weights_only=True)

    try:
        num_actions, hidden_size = weights["policy.actor.0.weight"].shape
        num_action_args, _ = weights["policy.actor.1.weight"].shape
        _, obs_channels, _, _ = weights["policy.network.0.weight"].shape
    except Exception as e:
        print(f"Failed automatic parse from weights: {e}")
        # TODO -- fix all magic numbers
        num_actions, num_action_args = 9, 10
        _, obs_channels = 128, 34

    # Create environment namespace
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[num_actions, num_action_args]),
        single_observation_space=SimpleNamespace(shape=tuple(torch.tensor([obs_channels, 11, 11]).tolist())),
    )

    if puffer is None:
        raise ValueError("Puffer config is required to load a Pytorch policy.")

    policy = instantiate(puffer, env=env, policy=None)
    policy.load_state_dict(weights)
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
        self.hidden_size = policy.hidden_size
        self.lstm = policy.lstm  # Point to the actual LSTM module, not the entire policy

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback for models with no parameters
            return torch.device("cpu")

    def forward(self, obs: torch.Tensor, state, action=None):
        """Uses variable names from LSTMWrapper. Translating for Metta:
        critic -> value
        logprob -> logprob_act
        hidden -> logits then, after sample_logits(), log_sftmx_logits
        """
        hidden, critic = self.policy(obs, state)  # using variable names from LSTMWrapper
        action, logprob, logits_entropy = sample_logits(hidden, action)
        return action, logprob, logits_entropy, critic, hidden

    def activate_actions(self, actions_names, actions_max_params, device):
        """
        Activates the action space for the Pytorch policy.
        This is a simple check to ensure the number of action heads matches.
        A more thorough check could compare action names and parameter counts if
        that metadata were available on the Pytorch model.
        """
        # Try to find the actor module in different locations
        actor = None
        if hasattr(self.policy, "actor"):
            actor = self.policy.actor
        elif hasattr(self.policy, "policy") and hasattr(self.policy.policy, "actor"):
            # Handle nested structure like Recurrent -> Policy -> actor
            actor = self.policy.policy.actor

        if actor is None:
            logging.warning("Could not find actor module in Pytorch model")
            return

        # Check if actor is a ModuleList or similar container
        if isinstance(actor, nn.ModuleList):
            num_action_heads = len(actor)
        elif isinstance(actor, nn.Module):
            # If it's a single module, assume 1 action head
            num_action_heads = 1
        else:
            logging.warning("Could not determine number of action heads in Pytorch model")
            return

        if num_action_heads != len(actions_max_params):
            logging.warning(
                f"Action space mismatch: Pytorch model has {num_action_heads} action heads, "
                f"but environment expects {len(actions_max_params)}. This may lead to errors."
            )
        else:
            logging.info(
                f"PytorchAgent action space activated with {num_action_heads} heads. "
                "No-op, as Pytorch models are not reconfigured at runtime."
            )

    def l2_reg_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def l2_init_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self):
        pass

    def clip_weights(self):
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        return []
