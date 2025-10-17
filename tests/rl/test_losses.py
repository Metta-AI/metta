"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.ppo import PPO


class DummyPolicy(Policy):
    """Minimal policy implementation for exercising loss utilities."""

    def __init__(self) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(1, 1)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        td = td.clone(False)
        td["values"] = torch.zeros(td.batch_size.numel(), dtype=torch.float32)
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return Composite(values=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32))

    def initialize_to_environment(self, game_rules, device: torch.device) -> None:  # noqa: D401
        return None

    @property
    def device(self) -> torch.device:  # noqa: D401
        return torch.device("cpu")

    @property
    def total_params(self) -> int:  # noqa: D401
        return sum(param.numel() for param in self.parameters())

    def reset_memory(self) -> None:  # noqa: D401
        return None

    def clip_weights(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=False)


class DummyLoss(Loss):
    """Loss subclass exposing the base-class helpers for testing."""

    def __init__(self) -> None:
        policy = DummyPolicy()
        trainer_cfg = SimpleNamespace()
        env = SimpleNamespace()
        loss_cfg = SimpleNamespace()
        super().__init__(policy, trainer_cfg, env, torch.device("cpu"), "dummy", loss_cfg)


def test_loss_stats_average_values() -> None:
    loss = DummyLoss()
    loss.loss_tracker["policy_loss"].extend([1.0, 3.0])
    loss.loss_tracker["value_loss"].extend([2.0, 4.0, 6.0])

    stats = loss.stats()

    assert stats["policy_loss"] == 2.0
    assert stats["value_loss"] == 4.0


def test_zero_loss_tracker_clears_values() -> None:
    loss = DummyLoss()
    loss.loss_tracker["entropy"].extend([0.1, 0.2])

    loss.zero_loss_tracker()

    assert all(len(values) == 0 for values in loss.loss_tracker.values())


def test_ppo_kl_divergence_matches_manual_computation() -> None:
    old_logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    new_logits = torch.tensor([[0.9, 0.1, -0.5], [0.3, 0.7, 0.0]], dtype=torch.float32)

    old_log_probs = torch.log_softmax(old_logits, dim=-1)
    new_log_probs = torch.log_softmax(new_logits, dim=-1)

    kl_from_helper = PPO._kl_divergence(old_log_probs, new_log_probs)
    manual_kl = torch.sum(old_log_probs.exp() * (old_log_probs - new_log_probs), dim=-1)

    assert torch.allclose(kl_from_helper, manual_kl, atol=1e-6)
