from __future__ import annotations

import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import torch

from metta.agent.policy import PolicyArchitecture
from metta.rl.policy_artifact import load_policy_artifact
from metta.rl.training import EnvironmentMetaData


@dataclass
class ParameterDifference:
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    expected_dtype: str
    actual_dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_shape": list(self.expected_shape),
            "actual_shape": list(self.actual_shape),
            "expected_dtype": self.expected_dtype,
            "actual_dtype": self.actual_dtype,
        }


@dataclass
class CheckpointCompatibilityReport:
    checkpoint_path: str
    success: bool = False
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: Dict[str, ParameterDifference] = field(default_factory=dict)
    dtype_mismatches: Dict[str, tuple[str, str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "success": self.success,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "shape_mismatches": {k: v.to_dict() for k, v in self.shape_mismatches.items()},
            "dtype_mismatches": {
                k: {"expected_dtype": v[0], "actual_dtype": v[1]} for k, v in self.dtype_mismatches.items()
            },
            "errors": self.errors,
            "notes": self.notes,
        }


def _ordered_state_dict(state: Mapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    if isinstance(state, OrderedDict):
        return state.copy()
    return OrderedDict((k, v) for k, v in state.items())


def check_checkpoint_compatibility(
    checkpoint_path: str | Path,
    *,
    policy_architecture: PolicyArchitecture,
    env_metadata: Optional[EnvironmentMetaData] = None,
    device: torch.device | None = None,
) -> CheckpointCompatibilityReport:
    """
    Compare an existing checkpoint against the provided policy architecture.

    Args:
        checkpoint_path: Path to the safetensors checkpoint.
        policy_architecture: Architecture to instantiate for comparison.
        env_metadata: Optional environment metadata required by the policy.
        device: Device to instantiate the policy on (defaults to CPU).
    """

    path = Path(checkpoint_path)
    report = CheckpointCompatibilityReport(checkpoint_path=str(path.resolve()))

    try:
        artifact = load_policy_artifact(path)
    except Exception as exc:  # pragma: no cover - defensive guard
        report.errors.append(_format_exception("Failed to load checkpoint", exc))
        return report

    if artifact.state_dict is None:
        if artifact.policy is not None:
            report.success = True
            report.notes.append("Checkpoint contains a serialized policy object; no state_dict to compare.")
            return report

        report.errors.append("Checkpoint does not contain weights.safetensors or policy.pt payloads.")
        return report

    try:
        policy = policy_architecture.make_policy(env_metadata)
    except Exception as exc:  # pragma: no cover - defensive guard
        report.errors.append(_format_exception("Failed to instantiate policy architecture", exc))
        return report

    target_device = device or torch.device("cpu")
    policy = policy.to(target_device)

    if hasattr(policy, "initialize_to_environment") and env_metadata is not None:
        try:
            policy.initialize_to_environment(env_metadata, target_device)
        except Exception as exc:  # pragma: no cover - defensive guard
            report.errors.append(_format_exception("initialize_to_environment failed", exc))

    target_state = _ordered_state_dict(policy.state_dict())
    source_state = _ordered_state_dict(artifact.state_dict)

    report.missing_keys = sorted(set(target_state.keys()) - set(source_state.keys()))
    report.unexpected_keys = sorted(set(source_state.keys()) - set(target_state.keys()))

    for key in _intersection(target_state.keys(), source_state.keys()):
        target_tensor = target_state[key]
        source_tensor = source_state[key]

        if not isinstance(target_tensor, torch.Tensor) or not isinstance(source_tensor, torch.Tensor):
            continue  # Skip non-tensor entries (buffers can be handled separately if needed)

        if target_tensor.shape != source_tensor.shape:
            report.shape_mismatches[key] = ParameterDifference(
                expected_shape=tuple(int(dim) for dim in target_tensor.shape),
                actual_shape=tuple(int(dim) for dim in source_tensor.shape),
                expected_dtype=str(target_tensor.dtype),
                actual_dtype=str(source_tensor.dtype),
            )

        if target_tensor.dtype != source_tensor.dtype:
            report.dtype_mismatches[key] = (str(target_tensor.dtype), str(source_tensor.dtype))

    if not (report.missing_keys or report.unexpected_keys or report.shape_mismatches or report.dtype_mismatches):
        try:
            policy.load_state_dict(source_state, strict=True)
            report.success = True
        except RuntimeError as exc:
            report.errors.append(_format_exception("load_state_dict(strict=True) failed", exc))
    else:
        report.success = False

    return report


def _format_exception(context: str, exc: Exception) -> str:
    return f"{context}: {exc}\n{traceback.format_exc()}"


def _intersection(left: Iterable[str], right: Iterable[str]) -> Iterable[str]:
    return sorted(set(left).intersection(right))
