"""Shared helpers for serializing policy checkpoints."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.training.training_environment import EnvironmentMetaData
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolicyArtifact:
    """Either a ready-to-use policy or metadata to rebuild one."""

    policy: Policy | None = None
    policy_architecture: PolicyArchitecture | None = None
    state_dict: dict[str, torch.Tensor] | None = None
    training_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        has_policy = self.policy is not None
        has_architecture = self.policy_architecture is not None
        has_state = self.state_dict is not None

        has_training_metrics = self.training_metrics is not None

        if has_policy:
            if has_architecture or has_state or has_training_metrics:
                raise ValueError(
                    "PolicyArtifact must be constructed with either a policy instance or"
                    " a policy_architecture + state_dict pair, not both."
                )
        else:
            if not (has_architecture and has_state):
                raise ValueError(
                    "PolicyArtifact requires a policy instance or both policy_architecture and state_dict."
                )
            if self.training_metrics is None:
                self.training_metrics = {}

    def instantiate(self, env_metadata: EnvironmentMetaData, *, strict: bool = True) -> Policy:
        """Return a policy, instantiating from metadata when needed."""

        if self.policy is not None:
            return self.policy

        assert self.policy_architecture is not None
        assert self.state_dict is not None

        policy = self.policy_architecture.make_policy(env_metadata)
        load_result = policy.load_state_dict(self.state_dict, strict=strict)
        if not strict and load_result is not None:
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing or unexpected:
                logger.warning(
                    "Loaded policy with missing keys: %s and unexpected keys: %s",
                    missing,
                    unexpected,
                )
        self.policy = policy
        return policy


def _load_policy_architecture(class_path: str, payload: dict) -> PolicyArchitecture:
    config_class = load_symbol(class_path)
    if not isinstance(config_class, type) or not issubclass(config_class, PolicyArchitecture):
        raise TypeError(f"Loaded symbol {class_path} is not a PolicyArchitecture subclass")

    if hasattr(config_class, "model_validate"):
        return config_class.model_validate(payload)  # type: ignore[attr-defined]
    if hasattr(config_class, "parse_obj"):
        return config_class.parse_obj(payload)  # type: ignore[attr-defined]
    return config_class(**payload)


def save_policy_artifact(
    *,
    base_path: str | Path,
    policy: Policy,
    policy_architecture: PolicyArchitecture,
    training_metrics: dict[str, Any] | None = None,
    detach_buffers: bool = True,
) -> tuple[Path, Path]:
    """Serialize policy weights and training metrics next to the given base path."""

    base = Path(base_path)
    weights_path = Path(f"{base}.safetensors")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict_iter: Iterable[tuple[str, torch.Tensor]] = policy.state_dict().items()
    if detach_buffers:
        state_dict_iter = (  # type: ignore[assignment]
            (key, tensor.detach().cpu()) for key, tensor in state_dict_iter
        )

    state_dict = OrderedDict(state_dict_iter)
    save_file(state_dict, str(weights_path))

    metrics_path = Path(f"{base}.metrics.json")
    metrics_path.write_text(
        json.dumps(training_metrics or {}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return weights_path, metrics_path


def load_policy_artifact(base_path: str | Path) -> PolicyArtifact:
    """Load policy weights and metrics serialized via :func:`save_policy_artifact`."""

    base = Path(base_path)
    weights_path = Path(f"{base}.safetensors")
    metrics_path = Path(f"{base}.metrics.json")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing policy weights file: {weights_path}")

    state_dict = OrderedDict(load_file(str(weights_path)).items())

    training_metrics = {}
    if metrics_path.exists():
        training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    manifest_path = base.parent / "model_architecture.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing policy architecture manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_class_path = manifest.get("class_path")
    config_data = manifest.get("config")
    if not config_class_path or config_data is None:
        raise ValueError("Invalid model architecture manifest; expected class_path and config fields")

    policy_architecture = _load_policy_architecture(config_class_path, config_data)

    return PolicyArtifact(
        policy_architecture=policy_architecture,
        state_dict=state_dict,
        training_metrics=training_metrics,
    )
