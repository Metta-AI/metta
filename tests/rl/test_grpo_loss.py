from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from tensordict import TensorDict

from metta.agent.policy import Policy
from metta.rl.loss.grpo import GRPO, GRPOConfig


class DummyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        batch = td.batch_size[0]
        td = td.clone(False)
        if action is None:
            action = torch.zeros((batch, 1), dtype=torch.long)
        else:
            if action.ndim == 1:
                action = action.unsqueeze(-1)
        td["actions"] = action
        td["act_log_prob"] = torch.zeros((batch, 1), dtype=torch.float32, requires_grad=True)
        td["entropy"] = torch.ones((batch, 1), dtype=torch.float32)
        td["values"] = torch.zeros((batch, 1), dtype=torch.float32)
        return td

    def initialize_to_environment(self, game_rules, device: torch.device) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset_memory(self) -> None:
        pass


def _make_env() -> SimpleNamespace:
    return SimpleNamespace(
        single_action_space=spaces.Discrete(2),
        single_observation_space=spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8),
    )


def test_grpo_config_creates_loss() -> None:
    cfg = GRPOConfig()
    policy = DummyPolicy()
    trainer_cfg = SimpleNamespace(total_timesteps=128, batch_size=32)
    env = _make_env()
    loss = cfg.create(policy, trainer_cfg, env, torch.device("cpu"), "grpo", cfg)
    assert isinstance(loss, GRPO)
    assert loss.loss_cfg.ratio_smoothing_beta == pytest.approx(0.2, rel=1e-6)


def test_grpo_process_minibatch_updates_trackers() -> None:
    cfg = GRPOConfig(ratio_smoothing_beta=0.5, gradient_penalty_coef=0.3, kl_penalty_coef=0.4, kl_target=0.05)
    policy = DummyPolicy()
    trainer_cfg = SimpleNamespace(total_timesteps=128, batch_size=32)
    env = _make_env()
    loss = cfg.create(policy, trainer_cfg, env, torch.device("cpu"), "grpo", cfg)
    loss.replay = SimpleNamespace(update=lambda *args, **kwargs: None)  # type: ignore[attr-defined]

    minibatch = TensorDict(
        {
            "act_log_prob": torch.zeros((2, 1), dtype=torch.float32),
            "values": torch.zeros((2, 1), dtype=torch.float32),
            "rewards": torch.ones((2, 1), dtype=torch.float32),
            "dones": torch.zeros((2, 1), dtype=torch.float32),
            "advantages": torch.tensor([[0.5], [0.2]], dtype=torch.float32),
            "returns": torch.ones((2, 1), dtype=torch.float32),
        },
        batch_size=(2, 1),
    )

    policy_td = TensorDict(
        {
            "act_log_prob": torch.full((2, 1), 0.1, dtype=torch.float32, requires_grad=True),
            "entropy": torch.ones((2, 1), dtype=torch.float32),
            "values": torch.zeros((2, 1), dtype=torch.float32),
        },
        batch_size=(2, 1),
    )

    indices = torch.arange(2, dtype=torch.long)
    prio_weights = torch.ones((2, 1), dtype=torch.float32)

    loss_value = loss._process_minibatch_update(minibatch, policy_td, indices, prio_weights)
    assert loss_value.requires_grad
    assert loss.loss_tracker["grpo_gradient_penalty"]
    assert loss.loss_tracker["grpo_kl_penalty"]
    assert loss.loss_tracker["grpo_ratio_smoothing"]
