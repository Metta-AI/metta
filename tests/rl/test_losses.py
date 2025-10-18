"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.stable_latent import StableLatentStateConfig


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


def _build_shared_td(latent: torch.Tensor, dones: torch.Tensor | None = None) -> TensorDict:
    """Helper to construct shared loss data structures."""
    segments, horizon, _ = latent.shape
    policy_td = TensorDict({"core": latent}, batch_size=[segments, horizon])
    mb_content: dict[str, torch.Tensor] = {}
    if dones is None:
        dones = torch.zeros(segments, horizon, 1, dtype=torch.bool)
    if dones.dim() == 2:
        dones = dones.unsqueeze(-1)
    mb_content["dones"] = dones
    minibatch = TensorDict(mb_content, batch_size=[segments, horizon])
    return TensorDict({"policy_td": policy_td, "sampled_mb": minibatch}, batch_size=[])


def test_stable_latent_state_loss_basic_penalty() -> None:
    cfg = StableLatentStateConfig(target_key="core", loss_coef=1.0)
    loss = cfg.create(DummyPolicy(), SimpleNamespace(), SimpleNamespace(), torch.device("cpu"), "stable", cfg)

    time_axis = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    latent = time_axis.repeat(2, 1, 3)
    shared = _build_shared_td(latent)

    value, *_ = loss.run_train(shared, SimpleNamespace(epoch=0), 0)

    assert value.item() == pytest.approx(1.0, rel=1e-5)
    assert loss.loss_tracker["stable_latent_loss"][-1] == pytest.approx(1.0, rel=1e-5)
    expected_delta = torch.sqrt(torch.tensor(3.0)).item()
    assert loss.loss_tracker["stable_latent_delta_l2"][-1] == pytest.approx(expected_delta, rel=1e-5)


def test_stable_latent_masks_episode_boundaries() -> None:
    cfg = StableLatentStateConfig(target_key="core", loss_coef=1.0)
    loss = cfg.create(DummyPolicy(), SimpleNamespace(), SimpleNamespace(), torch.device("cpu"), "stable", cfg)

    latent = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [10.0, 10.0],
                [11.0, 11.0],
            ]
        ],
        dtype=torch.float32,
    )
    dones = torch.tensor([[0, 1, 0, 0]], dtype=torch.bool)

    shared = _build_shared_td(latent, dones=dones)

    value, *_ = loss.run_train(shared, SimpleNamespace(epoch=0), 0)

    assert value.item() == pytest.approx(1.0, rel=1e-5)
    # Delta magnitude should ignore the large jump over the done boundary (sqrt(2) average).
    expected_delta = torch.sqrt(torch.tensor(2.0)).item()
    assert loss.loss_tracker["stable_latent_delta_l2"][-1] == pytest.approx(expected_delta, rel=1e-5)
