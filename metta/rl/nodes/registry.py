from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Iterable

from metta.rl.nodes.base import NodeConfig


@dataclass(frozen=True)
class NodeSpec:
    key: str
    config_cls: type[NodeConfig]
    default_enabled: bool
    has_rollout: bool = True
    has_train: bool = True
    writes_actions: bool = False
    produces_experience: bool = False
    rollout_requires: tuple[str, ...] = ()
    train_requires: tuple[str, ...] = ()


DEFAULT_NODE_ORDER: tuple[str, ...] = (
    "sliced_kickstarter",
    "sliced_scripted_cloner",
    "eer_kickstarter",
    "eer_cloner",
    "ppo_critic",
    "quantile_ppo_critic",
    "ppo_actor",
    "cmpo",
    "vit_reconstruction",
    "contrastive",
    "ema",
    "dynamics",
    "grpo",
    "supervisor",
    "sl_checkpointed_kickstarter",
    "kickstarter",
    "logit_kickstarter",
)

_NODE_SPECS_CACHE: list[NodeSpec] | None = None


def discover_node_specs() -> list[NodeSpec]:
    global _NODE_SPECS_CACHE
    if _NODE_SPECS_CACHE is not None:
        return list(_NODE_SPECS_CACHE)

    specs: list[NodeSpec] = []

    package = importlib.import_module("metta.rl.nodes")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module_info.name.endswith((".base", ".registry")):
            continue
        module = importlib.import_module(module_info.name)
        module_specs = getattr(module, "NODE_SPECS", None)
        if module_specs:
            specs.extend(module_specs)

    _NODE_SPECS_CACHE = _sort_specs(specs)
    _ensure_unique_keys(_NODE_SPECS_CACHE)
    return list(_NODE_SPECS_CACHE)


def node_specs_by_key() -> dict[str, NodeSpec]:
    return {spec.key: spec for spec in discover_node_specs()}


def default_nodes() -> dict[str, NodeConfig]:
    defaults: dict[str, NodeConfig] = {}
    for spec in discover_node_specs():
        defaults[spec.key] = spec.config_cls(enabled=spec.default_enabled)
    return defaults


def iter_enabled_specs(specs: Iterable[NodeSpec], node_cfgs: dict[str, NodeConfig]) -> list[NodeSpec]:
    return [spec for spec in specs if getattr(node_cfgs.get(spec.key), "enabled", False)]


def _sort_specs(specs: list[NodeSpec]) -> list[NodeSpec]:
    order_map = {name: idx for idx, name in enumerate(DEFAULT_NODE_ORDER)}
    indexed = list(enumerate(specs))
    fallback = len(order_map) + 100
    indexed.sort(key=lambda pair: (order_map.get(pair[1].key, fallback), pair[0]))
    return [spec for _, spec in indexed]


def _ensure_unique_keys(specs: list[NodeSpec]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in specs:
        if spec.key in seen:
            duplicates.append(spec.key)
        seen.add(spec.key)
    if duplicates:
        dup_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate node keys discovered: {dup_list}")
