"""Unit tests for the new loss infrastructure."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.cmpo import CMPOConfig
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


class DummyNode(NodeBase):
    """Node subclass exposing the base-class helpers for testing."""

    def __init__(self) -> None:
        policy = DummyPolicy()
        trainer_cfg = SimpleNamespace()
        env = SimpleNamespace()
        node_cfg = NodeConfig()
        super().__init__(policy, trainer_cfg, env, torch.device("cpu"), "dummy", node_cfg)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"values"}


def test_node_stats_average_values() -> None:
    node = DummyNode()
    node.loss_tracker["policy_loss"].extend([1.0, 3.0])
    node.loss_tracker["value_loss"].extend([2.0, 4.0, 6.0])

    stats = node.stats()

    assert stats["dummy/policy_loss"] == 2.0
    assert stats["dummy/value_loss"] == 4.0


def test_zero_loss_tracker_clears_values() -> None:
    node = DummyNode()
    node.loss_tracker["entropy"].extend([0.1, 0.2])

    node.zero_loss_tracker()

    assert all(len(values) == 0 for values in node.loss_tracker.values())


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
