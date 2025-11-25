"""Policy discovery and loading utilities."""

from __future__ import annotations

import functools
import importlib
import os
import pkgutil
import re
import urllib.parse
from pathlib import Path
from typing import Optional

import torch

from mettagrid.policy.artifact import load_policy_artifact, save_policy_artifact, save_policy_to_uri
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.util.module import load_symbol


def guess_data_dir() -> Path:
    """Return default data directory; kept here for callers that monkeypatch this module."""
    data_dir = os.environ.get("DATA_DIR")
    return Path(data_dir) if data_dir else Path("./train_dir")


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
    # Extract loader-only options (do not forward into policy constructors)
    device_override = device if device is not None else init_kwargs.pop("device", None)
    strict_override = init_kwargs.pop("strict", strict)
    state_dict: dict[str, torch.Tensor] | None = None
    architecture = arch_hint or init_kwargs.get("policy_architecture")
    artifact = None
    policy_from_artifact: MultiAgentPolicy | None = None
    checkpoint_ref = None

    if policy_spec.data_path:
        data_path = Path(policy_spec.data_path).expanduser()
        if data_path.exists():
            checkpoint_ref = str(data_path)

    if checkpoint_ref is None and init_kwargs.get("checkpoint_uri"):
        checkpoint_ref = str(init_kwargs["checkpoint_uri"])

    if checkpoint_ref:
        artifact = load_policy_artifact(checkpoint_ref)

    # Drop loader-only keys before constructing policies to avoid TypeError
    init_kwargs.pop("checkpoint_uri", None)
    init_kwargs.pop("policy_architecture", None)
    init_kwargs.pop("display_name", None)

    if artifact is not None:
        architecture = getattr(artifact, "policy_architecture", None) or architecture
        state_dict = getattr(artifact, "state_dict", None)
        policy_from_artifact = getattr(artifact, "policy", None)
        if (
            state_dict is not None
            and architecture is None
            and policy_from_artifact is None
            and not policy_spec.class_path
        ):
            raise ValueError("Loading checkpoints requires policy_architecture when none is embedded")

    if policy_from_artifact is not None:
        policy = policy_from_artifact
    elif architecture is not None and hasattr(architecture, "make_policy"):
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
        missing, unexpected = policy.load_state_dict(state_dict, strict=strict_override)
        if strict_override and (missing or unexpected):
            raise RuntimeError(f"Strict loading failed. Missing: {missing}, Unexpected: {unexpected}")

    if device_override is not None:
        policy = policy.to(torch.device(device_override))  # type: ignore[assignment]

    return policy



def initialize_or_load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_spec: PolicySpec,
    *,
    arch_hint: object | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> MultiAgentPolicy:
    """Wrapper that preserves legacy save_policy/display_name expectations."""
    policy = load_policy(
        policy_env_info,
        policy_spec,
        arch_hint=arch_hint,
        device=device,
        strict=strict,
    )

    display_name = (policy_spec.init_kwargs or {}).get("display_name") or policy_spec.class_path
    setattr(policy, "display_name", display_name)
    setattr(policy, "_display_name", display_name)

    def _save_policy(self, destination: str | Path, *, policy_architecture=None) -> str:
        arch = policy_architecture or getattr(self, "_policy_architecture", None) or arch_hint
        if arch is None:
            raise ValueError("policy_architecture is required to save policy")
        dest_str = str(destination)
        if dest_str.startswith("s3://"):
            from metta.rl.checkpoint_manager import write_file
            local_path = Path.cwd() / Path(dest_str).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            save_policy_artifact(local_path, policy_architecture=arch, state_dict=self.state_dict())
            write_file(dest_str, str(local_path))
            return dest_str
        return save_policy(destination, self, arch_hint=arch)

    policy.save_policy = _save_policy.__get__(policy, policy.__class__)  # type: ignore[attr-defined]
    return policy
initialize_or_load_policy = initialize_or_load_policy

def save_policy(
    destination: str | Path,
    policy: MultiAgentPolicy,
    *,
    arch_hint: object | None = None,
) -> str:
    """Persist a policy checkpoint to a local path or S3 via the shim."""
    dest = str(destination)
    suffix = Path(dest).suffix.lower()

    inner_policy = getattr(policy, "module", policy)
    state_dict = inner_policy.state_dict()
    architecture = arch_hint or getattr(inner_policy, "_policy_architecture", None)

    if suffix == ".mpt" and dest.startswith("s3://"):
        if architecture is None:
            raise ValueError("policy_architecture is required when saving .mpt")
        return save_policy_to_uri(dest, policy_architecture=architecture, state_dict=state_dict)

    path = Path(dest).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".mpt":
        if architecture is None:
            raise ValueError("policy_architecture is required when saving .mpt")
        save_policy_artifact(path, policy_architecture=architecture, state_dict=state_dict)
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
