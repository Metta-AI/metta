from __future__ import annotations

from typing import Any

import torch.nn as nn

from metta.agent.policy import Policy
from metta.rl.trainer import Trainer
from metta.rl.training import TrainingEnvironmentConfig
from metta.tools.train import HookSpec, TrainTool


class _CapturingPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.register_calls: list[tuple[str, str]] = []

    def forward(self, td, action=None):  # pragma: no cover - not used
        return td

    @property
    def device(self):  # pragma: no cover - not used
        raise NotImplementedError

    def reset_memory(self) -> None:  # pragma: no cover - not used
        pass

    def register_component_hook_rule(
        self,
        *,
        component_name: str,
        hook_factory,
        hook_type: str = "forward",
    ) -> None:
        self.register_calls.append((component_name, hook_type))
        module = nn.Linear(1, 1)
        handle = hook_factory(self, component_name, module)
        assert hasattr(handle, "remove")


def _spec(policy: Policy, trainer: Trainer) -> list[HookSpec]:
    def factory(policy: Policy, name: str, module: Any):
        return module.register_forward_hook(lambda *_: None)

    return [("component", factory, "forward")]


def test_add_training_hook_invokes_registered_hook() -> None:
    tool = TrainTool(training_env=TrainingEnvironmentConfig())
    tool.add_training_hook(_spec)

    policy = _CapturingPolicy()
    trainer = object()

    tool._run_training_hooks(policy=policy, trainer=trainer)  # type: ignore[arg-type]

    assert policy.register_calls == [("component", "forward")]
