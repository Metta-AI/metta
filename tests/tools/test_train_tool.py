from __future__ import annotations

from typing import Any

import torch

from metta.agent.policy import Policy
from metta.rl.trainer import Trainer
from metta.rl.training import TrainingEnvironmentConfig
from metta.tools.train import TrainTool


class _DummyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, td, action=None):  # pragma: no cover - not used
        return td

    @property
    def device(self):
        return torch.device("cpu")

    def reset_memory(self) -> None:
        pass


def test_add_training_hook_invokes_registered_hook() -> None:
    tool = TrainTool(training_env=TrainingEnvironmentConfig())
    calls: list[tuple[Policy, Trainer | Any]] = []

    def hook(policy: Policy, trainer: Trainer | Any) -> None:
        calls.append((policy, trainer))

    tool.add_training_hook(hook)

    sentinel_policy = _DummyPolicy()
    sentinel_trainer = object()

    tool._run_training_hooks(policy=sentinel_policy, trainer=sentinel_trainer)

    assert calls == [(sentinel_policy, sentinel_trainer)]
