"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.cmpo import CMPOConfig
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


def test_cmpo_adds_mirror_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CMPOConfig(mirror_coef=0.5)
    env = SimpleNamespace(single_action_space=SimpleNamespace(dtype=torch.float32))
    cmpo_loss = cfg.create(DummyPolicy(), SimpleNamespace(), env, torch.device("cpu"), "cmpo", cfg)

    def fake_super(
        self, minibatch: TensorDict, policy_td: TensorDict, indices: torch.Tensor, prio_weights: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(2.0, dtype=torch.float32, requires_grad=True)

    monkeypatch.setattr(PPO, "_process_minibatch_update", fake_super, raising=False)

    minibatch = TensorDict({"act_log_prob": torch.zeros(4, 1)}, batch_size=[4, 1])
    policy_td = TensorDict({"act_log_prob": torch.full((4, 1), 0.2)}, batch_size=[4, 1])
    indices = torch.zeros(4, dtype=torch.long)
    prio_weights = torch.ones(4, 1)

    loss_value = cmpo_loss._process_minibatch_update(minibatch, policy_td, indices, prio_weights)

    expected_penalty = 0.2**2
    assert loss_value.item() == pytest.approx(2.0 + cfg.mirror_coef * expected_penalty, rel=1e-5)
    assert cmpo_loss.loss_tracker["cmpo_mirror_penalty"][-1] == pytest.approx(expected_penalty, rel=1e-5)
