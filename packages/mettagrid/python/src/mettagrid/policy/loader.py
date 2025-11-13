"""Policy discovery and loading utilities."""

from __future__ import annotations

import functools
import importlib
import os
import pkgutil
import re
from pathlib import Path
from typing import Optional

from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.util.module import load_symbol


def initialize_or_load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
) -> MultiAgentPolicy:
    """Initialize a policy from its class path and optionally load weights.

    Args:
        policy_env_info: PolicyEnvInterface (created from env if not provided)
        policy_class_path: Full class path to the policy
        policy_data_path: Optional path to policy checkpoint

    Returns:
        Initialized policy instance
    """

    policy_class = load_symbol(policy_class_path)

    policy = policy_class(policy_env_info)  # type: ignore[misc]

    if policy_data_path:
        if not isinstance(policy, MultiAgentPolicy):
            raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

        policy.load_policy_data(policy_data_path)
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


def find_policy_checkpoints(checkpoints_path: Path, env_name: Optional[str] = None) -> list[Path]:
    checkpoints = []
    if env_name:
        # Try to find the final checkpoint
        # PufferLib saves checkpoints in data_dir/env_name/
        checkpoint_dir = checkpoints_path / env_name
        if checkpoint_dir.exists():
            checkpoints = checkpoint_dir.glob("*.pt")

    # Fallback: also check directly in checkpoints_path
    if not checkpoints and checkpoints_path.exists():
        checkpoints = checkpoints_path.glob("*.pt")
    return [
        p
        for p in sorted(checkpoints, key=lambda c: c.stat().st_mtime)
        if not any(re.fullmatch(pattern, p.name) for pattern in _NOT_CHECKPOINT_PATTERNS)
    ]


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
            raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
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
