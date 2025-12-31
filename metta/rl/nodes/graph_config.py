from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, model_validator

from metta.rl.nodes.base import NodeConfig
from metta.rl.nodes.registry import discover_node_specs, node_specs_by_key
from mettagrid.base_config import Config


def _default_nodes() -> dict[str, NodeConfig]:
    defaults: dict[str, NodeConfig] = {}
    for spec in discover_node_specs():
        defaults[spec.key] = spec.config_cls(enabled=spec.default_enabled)
    return defaults


class GraphConfig(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: dict[str, Any] = Field(default_factory=_default_nodes)

    @model_validator(mode="after")
    def _coerce_nodes(self) -> "GraphConfig":
        specs = node_specs_by_key()

        # Fill missing nodes with defaults
        for key, spec in specs.items():
            if key not in self.nodes:
                self.nodes[key] = spec.config_cls(enabled=spec.default_enabled)

        # Coerce provided entries into their config classes
        for key, value in list(self.nodes.items()):
            spec = specs.get(key)
            if spec is None:
                raise ValueError(f"Unknown node config '{key}'")
            if isinstance(value, spec.config_cls):
                continue
            self.nodes[key] = spec.config_cls.model_validate(value)

        return self
