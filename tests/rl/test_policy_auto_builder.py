from __future__ import annotations

from collections.abc import Callable

import pytest
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
        observation_space=None,
        action_space=None,
        feature_normalizations={},
    )


def _forward_hook_factory(events: list[str]) -> Callable[..., RemovableHandle]:
    def factory(policy, name: str, module) -> RemovableHandle:  # pragma: no cover - hook wiring
        return module.register_forward_hook(lambda *_: events.append(name))

    return factory


def _forward_pre_hook_factory(events: list[str]) -> Callable[..., RemovableHandle]:
    def factory(policy, name: str, module) -> RemovableHandle:  # pragma: no cover - hook wiring
        return module.register_forward_pre_hook(lambda *_: events.append(name))

    return factory


def test_register_forward_hook_rule_records_rule_and_handle() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook_factory=_forward_hook_factory(events),
        hook_type="forward",
    )

    assert "actor_mlp" in policy._hooks.forward_rules  # type: ignore[attr-defined]
    assert len(policy._hooks.forward_rules["actor_mlp"]) == 1  # type: ignore[attr-defined]
    handles = policy._hooks.forward_handles["actor_mlp"]  # type: ignore[attr-defined]
    assert len(handles) == 1
    assert isinstance(handles[0], RemovableHandle)

    handles[0].remove()


def test_register_forward_pre_hook_rule_records_rule_and_handle() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook_factory=_forward_pre_hook_factory(events),
        hook_type="forward_pre",
    )

    assert "actor_mlp" in policy._hooks.pre_rules  # type: ignore[attr-defined]
    assert len(policy._hooks.pre_rules["actor_mlp"]) == 1  # type: ignore[attr-defined]
    handles = policy._hooks.pre_handles["actor_mlp"]  # type: ignore[attr-defined]
    assert len(handles) == 1
    assert isinstance(handles[0], RemovableHandle)

    handles[0].remove()


def test_multiple_forward_hooks_are_appended() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())
    events: list[str] = []

    policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook_factory=_forward_hook_factory(events),
        hook_type="forward",
    )
    policy.register_component_hook_rule(
        component_name="actor_mlp",
        hook_factory=_forward_hook_factory(events),
        hook_type="forward",
    )

    rules = policy._hooks.forward_rules["actor_mlp"]  # type: ignore[attr-defined]
    handles = policy._hooks.forward_handles["actor_mlp"]  # type: ignore[attr-defined]

    assert len(rules) == 2
    assert len(handles) == 2
    assert handles[0] is not handles[1]

    for handle in handles:
        handle.remove()


def test_register_hook_missing_component_raises() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    with pytest.raises(KeyError):
        policy.register_component_hook_rule(
            component_name="does_not_exist",
            hook_factory=_forward_hook_factory([]),
            hook_type="forward",
        )


def test_attach_relu_activation_hooks_registers_forward_hook() -> None:
    policy = ViTDefaultConfig().make_policy(_game_rules())

    specs = attach_relu_activation_hooks(policy)
    assert specs

    for component_name, hook_factory, hook_type in specs:
        policy.register_component_hook_rule(
            component_name=component_name,
            hook_factory=hook_factory,
            hook_type=hook_type,
        )

    handles = policy._hooks.forward_handles.get("actor_mlp")  # type: ignore[attr-defined]
    assert handles is not None
    assert all(isinstance(handle, RemovableHandle) for handle in handles)

    assert get_relu_activation_metrics(policy) == {}
