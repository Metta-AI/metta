"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.loss.cmpo import CMPOConfig

try:
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym import spaces as gym_spaces  # type: ignore[no-redef]


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


def test_cmpo_config_initializes_world_model() -> None:
    cfg = CMPOConfig()
    env = SimpleNamespace(
        single_action_space=gym_spaces.Discrete(6),
        single_observation_space=gym_spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
    )
    trainer_cfg = SimpleNamespace(total_timesteps=1024, batch_size=64)

    cmpo_loss = cfg.create(DummyPolicy(), trainer_cfg, env, torch.device("cpu"), "cmpo", cfg)

    assert cmpo_loss.obs_dim == 4 * 4 * 3
    assert cmpo_loss.action_dim == 6
    assert len(cmpo_loss.world_model.members) == cfg.world_model.ensemble_size
    assert cmpo_loss.scheduler.current_ratio == cfg.scheduler.min_ratio
