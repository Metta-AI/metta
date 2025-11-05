from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.model_analysis import (
    FisherInformationState,
    SaturatedActivationState,
    attach_fisher_information_hooks,
    attach_relu_activation_hooks,
    attach_saturated_activation_hooks,
    get_relu_activation_metrics,
)
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


def test_register_forward_hook_rule_records_rule_and_handle() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
    )

    assert isinstance(handle, RemovableHandle)
    handle.remove()


def test_multiple_forward_hooks_are_appended() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    handle1 = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
    )
    handle2 = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=_forward_hook(events),
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
        )


def test_attach_relu_activation_hooks_registers_forward_hook() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    builder = attach_relu_activation_hooks()
    context = type("DummyContext", (), {"model_metrics": {}})()
    hook = builder("actor_mlp", context)
    assert hook is not None
    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=hook,
    )
    assert isinstance(handle, RemovableHandle)

    assert get_relu_activation_metrics(policy) == {}

    handle.remove()


def test_attach_fisher_information_hooks_registers_backward_hook() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    builder = attach_fisher_information_hooks()
    context = type("DummyContext", (), {"model_metrics": {}})()
    hook = builder("actor_mlp", context)
    assert hook is not None
    handle = policy.register_component_backward_hook_rule(
        component_name="actor_mlp",
        hook=hook,
    )
    assert isinstance(handle, RemovableHandle)

    # Access the state through the context
    state = context.model_metrics.get("fisher_information_state")
    assert isinstance(state, FisherInformationState)

    # Note: We can't easily test the backward hook without a full training setup,
    # but we've verified the hook is registered correctly.
    # The actual gradient accumulation will happen during training when backward() is called.

    handle.remove()


def test_attach_saturated_activation_hooks_registers_forward_hook() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    # Test tanh activation hooks
    builder = attach_saturated_activation_hooks(activation="tanh", extractor="actor_1")
    context = type("DummyContext", (), {"model_metrics": {}})()
    hook = builder("actor_mlp", context)
    assert hook is not None
    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=hook,
    )
    assert isinstance(handle, RemovableHandle)

    # Access the state through the context
    state = context.model_metrics.get("saturated_activation_state_tanh")
    assert isinstance(state, SaturatedActivationState)
    assert state.activation == "tanh"

    handle.remove()

    # Test sigmoid activation hooks
    builder = attach_saturated_activation_hooks(activation="sigmoid", extractor="actor_1")
    context = type("DummyContext", (), {"model_metrics": {}})()
    hook = builder("actor_mlp", context)
    assert hook is not None
    handle = policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook=hook,
    )
    assert isinstance(handle, RemovableHandle)

    state = context.model_metrics.get("saturated_activation_state_sigmoid")
    assert isinstance(state, SaturatedActivationState)
    assert state.activation == "sigmoid"

    handle.remove()
