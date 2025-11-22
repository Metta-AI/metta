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

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.util.module import load_symbol


def initialize_or_load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_spec: PolicySpec,
) -> MultiAgentPolicy:
    """Initialize a policy from its class path and optionally load weights.

    Expects PolicySpec to have local paths, shorthand or fully-specified. But should not have remote paths (e.g. s3://).

    Returns:
        Initialized policy instance
    """

    if policy_spec.data_path and policy_spec.data_path.lower().endswith(".mpt"):
        policy = _load_policy_artifact(policy_env_info, policy_spec)
        if not isinstance(policy, MultiAgentPolicy):
            raise TypeError("Loaded policy artifact did not produce a MultiAgentPolicy")
        return policy

    policy_class = load_symbol(resolve_policy_class_path(policy_spec.class_path))

    try:
        policy = policy_class(policy_env_info, **(policy_spec.init_kwargs or {}))  # type: ignore[call-arg]
    except TypeError as e:
        raise TypeError(
            f"Failed initializing policy {policy_spec.class_path} with kwargs {policy_spec.init_kwargs}: {e}"
        ) from e

    if policy_spec.data_path:
        policy.load_policy_data(policy_spec.data_path)

    if not isinstance(policy, MultiAgentPolicy):
        if isinstance(policy, AgentPolicy):
            raise TypeError(
                f"Policy {policy_spec.class_path} is an AgentPolicy, but should be a MultiAgentPolicy "
                f"(which returns AgentPolicy via `agent_policy`)"
            )
        raise TypeError(f"Policy {policy_spec.class_path} is not a MultiAgentPolicy")

    return policy


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


def _load_policy_artifact(
    policy_env_info: PolicyEnvInterface,
    policy_spec: PolicySpec,
) -> MultiAgentPolicy:
    """Load a policy from a .mpt artifact."""

    data_path = policy_spec.data_path
    if data_path is None:
        raise ValueError("data_path is required to load a policy artifact")

    init_kwargs = policy_spec.init_kwargs or {}
    device_arg = init_kwargs.get("device", "cpu")
    device = device_arg if isinstance(device_arg, torch.device) else torch.device(device_arg)
    strict = bool(init_kwargs.get("strict", True))

    try:
        from metta.rl.policy_artifact import load_policy_artifact
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Loading .mpt checkpoints requires the metta RL components to be installed.") from exc

    if "://" in data_path:
        try:
            from metta.rl.checkpoint_manager import CheckpointManager
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Loading checkpoint URIs requires the metta RL components to be installed.") from exc
        artifact = CheckpointManager.load_artifact_from_uri(data_path)
    else:
        artifact = load_policy_artifact(Path(data_path))

    if artifact.policy_architecture is None and artifact.state_dict is not None:
        arch_hint = policy_spec.init_kwargs.get("policy_architecture") if policy_spec.init_kwargs else None
        if arch_hint is not None:
            artifact.policy_architecture = arch_hint  # type: ignore[assignment]
        else:
            msg = (
                "Checkpoint contains weights but no policy_architecture; provide one via "
                "init_kwargs.policy_architecture"
            )
            raise ValueError(msg)

    return artifact.instantiate(policy_env_info, device=device, strict=strict)


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
