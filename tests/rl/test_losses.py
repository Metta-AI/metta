"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.cmpo import CMPOConfig
from metta.rl.loss.loss import Loss
from metta.rl.loss.stable_latent import StableLatentStateConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

try:
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym import spaces as gym_spaces  # type: ignore[no-redef]


class DummyPolicy(Policy):
    """Minimal policy implementation for exercising loss utilities."""

    def __init__(self, policy_env_info: PolicyEnvInterface | None = None) -> None:
        if policy_env_info is None:
            from mettagrid.config.mettagrid_config import MettaGridConfig

            policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)
        self._linear = torch.nn.Linear(1, 1)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        td = td.clone(False)
        td["values"] = torch.zeros(td.batch_size.numel(), dtype=torch.float32)
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return Composite(values=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32))

    def initialize_to_environment(self, policy_env_info, device: torch.device) -> None:  # noqa: D401
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

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"values"}


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


@pytest.fixture
def stable_latent_loss() -> Loss:
    cfg = StableLatentStateConfig(target_key="core", loss_coef=1.0)
    return cfg.create(DummyPolicy(), SimpleNamespace(), SimpleNamespace(), torch.device("cpu"), "stable")


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


def test_stable_latent_state_loss_basic_penalty(stable_latent_loss: Loss) -> None:
    time_axis = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    latent = time_axis.repeat(2, 1, 3)
    shared = _build_shared_td(latent)

    value, *_ = stable_latent_loss.run_train(shared, SimpleNamespace(epoch=0), 0)

    assert value.item() == pytest.approx(1.0, rel=1e-5)
    assert stable_latent_loss.loss_tracker["stable_latent_loss"][-1] == pytest.approx(1.0, rel=1e-5)
    expected_delta = torch.sqrt(torch.tensor(3.0)).item()
    assert stable_latent_loss.loss_tracker["stable_latent_delta_l2"][-1] == pytest.approx(expected_delta, rel=1e-5)


def test_stable_latent_masks_episode_boundaries(stable_latent_loss: Loss) -> None:
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

    value, *_ = stable_latent_loss.run_train(shared, SimpleNamespace(epoch=0), 0)

    assert value.item() == pytest.approx(1.0, rel=1e-5)
    # Delta magnitude should ignore the large jump over the done boundary (sqrt(2) average).
    expected_delta = torch.sqrt(torch.tensor(2.0)).item()
    assert stable_latent_loss.loss_tracker["stable_latent_delta_l2"][-1] == pytest.approx(expected_delta, rel=1e-5)


def test_cmpo_config_initializes_world_model() -> None:
    cfg = CMPOConfig()
    env = SimpleNamespace(
        single_action_space=gym_spaces.Discrete(6),
        single_observation_space=gym_spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
    )
    trainer_cfg = SimpleNamespace(
        total_timesteps=1024,
        batch_size=64,
        advantage=SimpleNamespace(gamma=0.99, gae_lambda=0.95, vtrace_rho_clip=1.0, vtrace_c_clip=1.0),
    )

    cmpo_loss = cfg.create(DummyPolicy(), trainer_cfg, env, torch.device("cpu"), "cmpo")

    assert cmpo_loss.obs_dim == 4 * 4 * 3
    assert cmpo_loss.action_dim == 6
    assert len(cmpo_loss.world_model.members) == cfg.world_model.ensemble_size
    assert cmpo_loss.prior_model is None
