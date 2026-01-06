from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from metta.rl.nodes.base import NodeConfig


@dataclass(frozen=True)
class NodeSpec:
    key: str
    config_cls: type[NodeConfig]
    default_enabled: bool
    has_rollout: bool = True
    has_train: bool = True


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


@lru_cache(maxsize=1)
def _discover_node_specs_cached() -> tuple[NodeSpec, ...]:
    specs: list[NodeSpec] = []

    package = importlib.import_module("metta.rl.nodes")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module_info.name.endswith((".base", ".registry")):
            continue
        module = importlib.import_module(module_info.name)
        module_specs = getattr(module, "NODE_SPECS", None)
        if module_specs:
            specs.extend(module_specs)

    specs = _sort_specs(specs)
    _ensure_unique_keys(specs)
    return tuple(specs)


def discover_node_specs() -> list[NodeSpec]:
    return list(_discover_node_specs_cached())


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
    # Use a high fallback value for nodes not in the default order
    UNORDERED_NODE_PRIORITY = len(order_map) + 100
    indexed.sort(key=lambda pair: (order_map.get(pair[1].key, UNORDERED_NODE_PRIORITY), pair[0]))
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
