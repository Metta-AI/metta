from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pufferlib.pytorch import sample_logits
from torch import nn


def load_policy(path: str, device: str = "cpu", puffer: DictConfig = None):
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

    def forward(self, obs: torch.Tensor, state, action=None):
        """Translates legacy policy output to MettaAgent interface.

        Legacy policies return: (hidden, critic)
        MettaAgent expects: (action, action_log_prob, entropy, value, log_probs)
        """
        # Get raw outputs from legacy policy
        hidden, critic = self.policy(obs, state)

        # Use pufferlib's sample_logits to handle action sampling
        action, logprob, logits_entropy = sample_logits(hidden, action)

        # Return in MettaAgent format
        # hidden -> log_probs (the raw logits)
        # critic -> value
        return action, logprob, logits_entropy, critic, hidden
