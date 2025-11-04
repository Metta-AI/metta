from __future__ import annotations

import types
from types import SimpleNamespace

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.trainer import Trainer
from metta.rl.training import GameRules, TrainingEnvironmentConfig
from metta.tools.train import TrainTool


def _game_rules() -> GameRules:
    return GameRules(
        obs_width=1,
        obs_height=1,
        obs_features={},
        action_names=["noop"],
        num_agents=1,
        observation_space=SimpleNamespace(shape=(1,)),
        action_space=SimpleNamespace(n=1),
        feature_normalizations={},
    )


def _builder(component_name: str, trainer: Trainer):
    if component_name != "actor_mlp":
        return None

    def hook(*_args) -> None:
        return None

    return hook


def test_add_training_hook_invokes_registered_hook() -> None:
    tool = TrainTool(training_env=TrainingEnvironmentConfig())
    tool.add_training_hook("actor_mlp", _builder)

    policy = ViTDefaultConfig().make_policy(_game_rules())
    trainer = type("DummyTrainer", (), {})()

    calls: list[tuple[str, str]] = []
    original_register = policy.register_component_hook_rule

    def tracking_register(self, *, component_name: str, hook, hook_type: str = "forward"):
        calls.append((component_name, hook_type))
        return original_register(component_name=component_name, hook=hook, hook_type=hook_type)

    policy.register_component_hook_rule = types.MethodType(tracking_register, policy)

    tool._register_policy_hooks(policy=policy, trainer=trainer)

    assert calls == [("actor_mlp", "forward")]
    assert len(tool._active_policy_hooks) == 1  # type: ignore[attr-defined]

    tool._clear_policy_hooks()  # type: ignore[attr-defined]
    assert not tool._active_policy_hooks  # type: ignore[attr-defined]
