from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.model_analysis import attach_relu_activation_hooks, get_relu_activation_metrics
from metta.rl.training import GameRules


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


def _forward_hook(events: list[str]) -> Callable[..., None]:
    def hook(module: nn.Module, *_args: Any) -> None:
        events.append(module.__class__.__name__)

    return hook


def _forward_pre_hook(events: list[str]) -> Callable[..., None]:
    def hook(module: nn.Module, *_args: Any) -> None:
        events.append(module.__class__.__name__)

    return hook


def test_register_forward_hook_rule_records_rule_and_handle() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
        hook_type="forward",
    )

    assert isinstance(handle, RemovableHandle)
    handle.remove()


def test_register_forward_pre_hook_rule_records_rule_and_handle() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_pre_hook(events),
        hook_type="forward_pre",
    )

    assert isinstance(handle, RemovableHandle)
    handle.remove()


def test_multiple_forward_hooks_are_appended() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    handle1 = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
        hook_type="forward",
    )
    handle2 = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
        hook_type="forward",
    )

    assert isinstance(handle1, RemovableHandle)
    assert isinstance(handle2, RemovableHandle)
    assert handle1 is not handle2
    handle1.remove()
    handle2.remove()


def test_register_hook_missing_component_raises() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    with pytest.raises(KeyError):
        policy.register_component_hook_rule(
            component_name="does_not_exist",
            hook=_forward_hook([]),
            hook_type="forward",
        )


def test_attach_relu_activation_hooks_registers_forward_hook() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    builder = attach_relu_activation_hooks()
    trainer = type("DummyTrainer", (), {})()
    hook = builder("actor_mlp", trainer)
    assert hook is not None
    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=hook,
        hook_type="forward",
    )
    assert isinstance(handle, RemovableHandle)

    assert get_relu_activation_metrics(policy) == {}

    handle.remove()
