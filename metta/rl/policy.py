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
        logger.info(
            f"Successfully parsed model architecture from weights: actions={num_actions}, args={num_action_args}, channels={obs_channels}"
        )
    except Exception as e:
        logger.warning(f"Failed automatic parse from weights: {e}")
        # Try alternative weight keys
        try:
            # Check for alternative naming conventions
            if "actor.0.weight" in weights:
                num_actions, hidden_size = weights["actor.0.weight"].shape
                num_action_args, _ = weights["actor.1.weight"].shape
                _, obs_channels, _, _ = weights["network.0.weight"].shape
                logger.info(
                    f"Parsed using alternative naming: actions={num_actions}, args={num_action_args}, channels={obs_channels}"
                )
            else:
                raise KeyError("No recognized weight naming pattern found")
        except Exception as e2:
            logger.warning(f"Alternative parsing also failed: {e2}")
            # TODO -- fix all magic numbers
            num_actions, num_action_args = 9, 10
            _, obs_channels = 128, 34
            logger.warning(
                f"Using fallback values: actions={num_actions}, args={num_action_args}, channels={obs_channels}"
            )

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
        # Initialize LSTM state if None
        if state.lstm_h is None or state.lstm_c is None:
            batch_size = obs.shape[0]
            state.lstm_h = torch.zeros(batch_size, self.hidden_size, device=obs.device)
            state.lstm_c = torch.zeros(batch_size, self.hidden_size, device=obs.device)

        hidden, critic = self.policy(obs, state)  # using variable names from LSTMWrapper
        action, logprob, logits_entropy = sample_logits(hidden, action)
        return action, logprob, logits_entropy, critic, hidden

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """PyTorch agents handle their own action spaces, so this is a no-op."""
        logger.info(f"PytorchAgent received action activation request for {len(action_names)} actions")

    def l2_reg_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def l2_init_loss(self) -> torch.Tensor:
        """Return zero loss for PyTorch agents."""
        return torch.zeros(1, device=self.device)

    def update_l2_init_weight_copy(self):
        """No-op for PyTorch agents."""
        pass

    def clip_weights(self):
        """No-op for PyTorch agents."""
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Return empty metrics for PyTorch agents."""
        return []
