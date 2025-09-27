"""Softmax foundational library.

This package centralizes shared utilities, test helpers, and compatibility
shims for the legacy ``metta`` namespace.  New code should import from
``softmax.lib`` while existing ``metta`` imports continue to function via
forwarders.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Final, Iterable

from . import test_support, tests_support, utils

__all__ = ["utils", "tests_support", "test_support"]

# Map well-known ``metta.common`` submodules into the ``softmax.lib`` namespace
# so new imports (``from softmax.lib.util.constants import ...``) resolve to the
# existing implementations until those modules migrate into this package.
_COMPAT_MODULES: Final[dict[str, str]] = {
    "datadog": "metta.common.datadog",
    "datadog.tracing": "metta.common.datadog.tracing",
    "silence_warnings": "metta.common.silence_warnings",
    "tool": "metta.common.tool",
    "util": "metta.common.util",
    "util.collections": "metta.common.util.collections",
    "util.constants": "metta.common.util.constants",
    "util.fs": "metta.common.util.fs",
    "util.log_config": "metta.common.util.log_config",
    "util.heartbeat": "metta.common.util.heartbeat",
    "util.retry": "metta.common.util.retry",
    "util.numpy_helpers": "metta.common.util.numpy_helpers",
    "wandb": "metta.common.wandb",
    "wandb.context": "metta.common.wandb.context",
    "wandb.utils": "metta.common.wandb.utils",
}


def _install_module_aliases(modules: Iterable[tuple[str, str]]) -> None:
    for alias, target in modules:
        if f"{__name__}.{alias}" in sys.modules:
            # Respect explicitly defined modules such as ``softmax.lib.utils``.
            continue
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError:  # pragma: no cover - defensive guard
            continue
        sys.modules[f"{__name__}.{alias}"] = module


_install_module_aliases(_COMPAT_MODULES.items())


def _load_metta_common() -> ModuleType:
    """Return the root ``metta.common`` module for attribute fallbacks."""

    return importlib.import_module("metta.common")


def __getattr__(name: str) -> ModuleType:
    """Expose attributes from ``metta.common`` for any remaining lookups."""

    module = _load_metta_common()
    return getattr(module, name)


def __dir__() -> list[str]:  # pragma: no cover - mirrors module spec
    base_names = set(__all__)
    base_names.update(_COMPAT_MODULES.keys())
    base_names.update(dir(_load_metta_common()))
    return sorted(base_names)
