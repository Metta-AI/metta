"""Policy discovery and loading utilities."""

from __future__ import annotations

import functools
import importlib
import os
import pkgutil
from typing import Optional

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.util.module import load_symbol


def initialize_or_load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_spec: PolicySpec,
    device_override: str | None = None,
) -> MultiAgentPolicy:
    """Initialize a policy from its class path and optionally load weights.

    Expects PolicySpec to have local paths, shorthand or fully-specified. But should not have remote paths (e.g. s3://).

    Returns:
        Initialized policy instance
    """

    kwargs = policy_spec.init_kwargs or {}
    if device_override is not None and "device" in kwargs:
        kwargs["device"] = device_override
    policy_class = load_symbol(resolve_policy_class_path(policy_spec.class_path))
    # We're planning to remove kwargs from the policy spec, maybe in January
    # 2026. We may want to support passing arguments, but they shouldn't take
    # the form of arbitrary kwargs where the policy author and our execution
    # code need to share a namespace.
    policy = policy_class(policy_env_info, **kwargs)  # type: ignore[call-arg]

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
    return registry.get(policy, policy)


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    """Get the shorthand name for a policy class path.

    Args:
        policy: Full class path to the policy.

    Returns:
        The shorthand name if found, None otherwise.
    """
    registry = get_policy_registry()
    return {v: k for k, v in registry.items()}.get(policy)


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

    # Check all paths (packages can have multiple paths)
    for path in package_path:
        # Use iter_modules to find modules and packages
        for _finder, name, ispkg in pkgutil.iter_modules([path], package_name + "."):
            if ".bindings" in name:
                continue
            try:
                importlib.import_module(name)
                # If it's a package, recursively discover its submodules
                if ispkg:
                    _walk_and_import_package(name)
            except (ImportError, AttributeError, TypeError, OSError):
                # Skip modules that can't be imported (may have missing dependencies)
                # OSError covers ctypes failing to load native libraries (e.g., Nim bindings)
                pass

        # Also check for namespace packages (directories without __init__.py)
        # that might not be found by iter_modules
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith("__") and not item.startswith("."):
                    namespace_name = f"{package_name}.{item}"
                    if ".bindings" in namespace_name:
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
