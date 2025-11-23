"""Policy discovery and loading utilities."""

from __future__ import annotations

import functools
import importlib
import os
import pkgutil
import re
from pathlib import Path
from typing import Optional

import torch
import urllib.parse

from mettagrid.policy.artifact import load_policy_artifact, save_policy_artifact_safetensors
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.util.module import load_symbol


def _unwrap_state_dict(payload: object) -> tuple[dict[str, torch.Tensor], object | None]:
    """Extract a state_dict and optional architecture from torch load payloads."""
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict") or payload.get("weights") or payload
        arch = (
            payload.get("policy_architecture")
            or payload.get("policy_architecture_spec")
            or payload.get("policy_architecture_str")
        )
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict payload must be a mapping, got {type(state_dict)}")
        return dict(state_dict), arch  # type: ignore[arg-type]
    raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload)}")


def load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_spec: PolicySpec,
    *,
    arch_hint: object | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> MultiAgentPolicy:
    """Initialize a policy from its spec and load weights if provided."""

    init_kwargs = dict(policy_spec.init_kwargs or {})
    state_dict: dict[str, torch.Tensor] | None = None
    architecture = arch_hint or init_kwargs.get("policy_architecture")

    if policy_spec.data_path:
        data_path = Path(policy_spec.data_path).expanduser()
        suffix = data_path.suffix.lower()
        if suffix == ".mpt":
            artifact = load_policy_artifact(data_path)
            architecture = getattr(artifact, "policy_architecture", None) or architecture
            state_dict = getattr(artifact, "state_dict", None)
            if state_dict is not None and architecture is None:
                raise ValueError("Old-format .mpt requires policy_architecture (provide arch_hint)")
        elif suffix == ".pt":
            payload = torch.load(data_path, map_location="cpu", weights_only=False)
            state_dict, arch_value = _unwrap_state_dict(payload)
            if arch_value is not None:
                architecture = arch_value
            if architecture is None:
                raise ValueError("Loading .pt requires policy_architecture when none is embedded")
        else:
            raise ValueError(f"Unsupported checkpoint extension: {suffix}")

    if architecture is not None and hasattr(architecture, "make_policy"):
        policy = architecture.make_policy(policy_env_info)  # type: ignore[call-arg]
    else:
        policy_class = load_symbol(resolve_policy_class_path(policy_spec.class_path))
        try:
            policy = policy_class(policy_env_info, **init_kwargs)  # type: ignore[call-arg]
        except TypeError as e:
            raise TypeError(
                f"Failed initializing policy {policy_spec.class_path} with kwargs {policy_spec.init_kwargs}: {e}"
            ) from e

    if not isinstance(policy, MultiAgentPolicy):
        if isinstance(policy, AgentPolicy):
            raise TypeError(
                f"Policy {policy_spec.class_path} is an AgentPolicy, but should be a MultiAgentPolicy "
                f"(which returns AgentPolicy via `agent_policy`)"
            )
        raise TypeError(f"Policy {policy_spec.class_path} is not a MultiAgentPolicy")

    if state_dict is not None:
        missing, unexpected = policy.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

    if device is not None:
        policy = policy.to(torch.device(device))  # type: ignore[assignment]

    return policy


# Alias retained for callers familiar with the old name
initialize_or_load_policy = load_policy


initialize_or_load_policy = load_policy


def save_policy(
    destination: str | Path,
    policy: MultiAgentPolicy,
    *,
    arch_hint: object | None = None,
) -> str:
    """Persist a policy checkpoint to a local path."""
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    inner_policy = getattr(policy, "module", policy)
    state_dict = inner_policy.state_dict()

    if suffix == ".mpt":
        architecture = arch_hint or getattr(inner_policy, "_policy_architecture", None)
        if architecture is None:
            raise ValueError("policy_architecture is required when saving .mpt")
        save_policy_artifact_safetensors(path, policy_architecture=architecture, state_dict=state_dict)
    elif suffix == ".pt":
        torch.save(state_dict, path)
    else:
        raise ValueError(f"Unsupported checkpoint extension: {suffix}")

    uri = f"file://{path.resolve()}"
    return urllib.parse.unquote(uri)


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "stateless", "token", "lstm" or a full class path.

    Returns:
        Full class path to the policy.
    """
    discover_and_register_policies()
    registry = get_policy_registry()
    full_path = registry.get(policy, policy)

    # Will raise an error if invalid
    _ = load_symbol(full_path)
    return full_path


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    """Get the shorthand name for a policy class path.

    Args:
        policy: Full class path to the policy.

    Returns:
        The shorthand name if found, None otherwise.
    """
    registry = get_policy_registry()
    return {v: k for k, v in registry.items()}.get(policy)


_NOT_CHECKPOINT_PATTERNS = (
    r"trainer_state\.pt",  # trainer state file
    r"model_\d{6}\.pt",  # matches model_000001.pt etc
)

_CHECKPOINT_GLOBS = ("*.pt", "*.mpt")


def find_policy_checkpoints(checkpoints_path: Path, env_name: Optional[str] = None) -> list[Path]:
    def _collect(path: Path) -> list[Path]:
        matches: list[Path] = []
        for pattern in _CHECKPOINT_GLOBS:
            matches.extend(path.glob(pattern))
        return matches

    checkpoints: list[Path] = []
    if env_name:
        checkpoint_dir = checkpoints_path / env_name
        if checkpoint_dir.exists():
            checkpoints = _collect(checkpoint_dir)

    if not checkpoints and checkpoints_path.exists():
        checkpoints = _collect(checkpoints_path)

    filtered = [
        p for p in checkpoints if not any(re.fullmatch(pattern, p.name) for pattern in _NOT_CHECKPOINT_PATTERNS)
    ]

    return sorted(filtered, key=lambda c: c.stat().st_mtime)


def resolve_policy_data_path(
    policy_data_path: Optional[str],
) -> Optional[str]:
    """Resolve a checkpoint path if provided.

    If the supplied path does not exist locally and AWS policy storage is configured,
    this will attempt to download the checkpoint into the requested location.
    """

    if policy_data_path is None:
        return None

    path = Path(policy_data_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        checkpoints = find_policy_checkpoints(path)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files (*.pt/*.mpt) found in directory: {path}")
        return str(checkpoints[-1])

    if path.exists():  # Non-pt extension but present
        return str(path)

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


@functools.cache
def _walk_and_import_package(package_name: str) -> None:
    """Discover and import all modules in a policy package to trigger registration.

    Args:
        package_name: Name of the package to scan (e.g., "mettagrid.policy")
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        # Package not installed, skip
        return

    # Walk through all submodules recursively
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    def _should_skip(module_name: str) -> bool:
        return ".bindings" in module_name

    # Check all paths (packages can have multiple paths)
    for path in package_path:
        # Use iter_modules to find modules and packages
        for _finder, name, ispkg in pkgutil.iter_modules([path], package_name + "."):
            if _should_skip(name):
                continue
            try:
                importlib.import_module(name)
                # If it's a package, recursively discover its submodules
                if ispkg:
                    _walk_and_import_package(name)
            except (ImportError, AttributeError, TypeError):
                # Skip modules that can't be imported (may have missing dependencies)
                pass

        # Also check for namespace packages (directories without __init__.py)
        # that might not be found by iter_modules
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith("__") and not item.startswith("."):
                    namespace_name = f"{package_name}.{item}"
                    if _should_skip(namespace_name):
                        continue
                    # Try to import as a namespace package
                    try:
                        importlib.import_module(namespace_name)
                        # Recursively discover its submodules
                        _walk_and_import_package(namespace_name)
                    except ImportError:
                        # Not a valid namespace package, skip
                        pass
        except OSError:
            # Can't list directory, skip
            pass


# Discover and import policy modules from all policy packages
# This allows policies to register themselves without creating hard dependencies
def discover_and_register_policies(*packages: str) -> None:
    for package_name in ["mettagrid.policy", "metta.agent.policy", "cogames.policy", *packages]:
        _walk_and_import_package(package_name)
