from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from tensordict import TensorDict

from metta.agent.policy import Policy
from metta.rl.loss.imitation import ImitationConfig


class ToyPolicy(Policy):
    def __init__(self, obs_dim: int, num_actions: int) -> None:
        super().__init__()
        self._obs_dim = obs_dim
        self._num_actions = num_actions
        self._linear = nn.Linear(obs_dim, num_actions)
        nn.init.uniform_(self._linear.weight, -0.01, 0.01)
        nn.init.zeros_(self._linear.bias)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        obs = td["env_obs"].to(dtype=torch.float32).view(td.batch_size[0], -1)
        logits = self._linear(obs)
        log_probs = torch.log_softmax(logits, dim=-1)

        if action is None:
            action = torch.argmax(log_probs, dim=-1, keepdim=True)
        else:
            action = action.to(dtype=torch.long)
            if action.ndim == 1:
                action = action.unsqueeze(-1)

        td = td.clone(False)
        td["actions"] = action
        td["act_log_prob"] = log_probs.gather(1, action)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1, keepdim=True)
        td["entropy"] = entropy
        td["values"] = torch.zeros_like(td["act_log_prob"])
        return td

    def initialize_to_environment(self, game_rules, device: torch.device) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset_memory(self) -> None:
        pass


def _write_demo(tmp_path: Path, obs_dim: int, num_actions: int) -> Path:
    obs = torch.randint(0, 255, (128, obs_dim), dtype=torch.uint8)
    actions = torch.randint(0, num_actions, (128,), dtype=torch.long)
    path = tmp_path / "demo.pt"
    torch.save({"env_obs": obs, "actions": actions}, path)
    return path


def test_imitation_loss_generates_grad(tmp_path: Path) -> None:
    obs_dim = 6
    num_actions = 4
    dataset_path = _write_demo(tmp_path, obs_dim, num_actions)

    policy = ToyPolicy(obs_dim=obs_dim, num_actions=num_actions)
    trainer_cfg = SimpleNamespace(total_timesteps=128, batch_size=32)
    env = SimpleNamespace(
        single_observation_space=spaces.Box(low=0, high=255, shape=(obs_dim,), dtype=np.uint8),
        single_action_space=spaces.Discrete(num_actions),
    )

    cfg = ImitationConfig(dataset_path=str(dataset_path), batch_size=32, loss_coef=1.0)
    loss = cfg.create(policy, trainer_cfg, env, torch.device("cpu"), "imitation", cfg)

    shared = TensorDict({}, batch_size=())
    context = SimpleNamespace(epoch=0)
    loss_value, _, stop = loss.run_train(shared, context, 0)

    assert not stop
    assert loss_value.requires_grad
    assert loss_value.item() > 0

    loss_value.backward()
    grad_norm = policy._linear.weight.grad.norm().item()
    assert grad_norm > 0


def test_imitation_loss_raises_for_missing_dataset(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    cfg = ImitationConfig(dataset_path=str(missing))
    policy = ToyPolicy(obs_dim=4, num_actions=2)
    trainer_cfg = SimpleNamespace(total_timesteps=1, batch_size=1)
    env = SimpleNamespace(
        single_observation_space=spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
        single_action_space=spaces.Discrete(2),
    )

    with pytest.raises(FileNotFoundError, match="Imitation dataset not found"):
        cfg.create(policy, trainer_cfg, env, torch.device("cpu"), "imitation", cfg)
