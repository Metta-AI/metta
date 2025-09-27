"""Bootstrapping helpers for `softmax.config` setup factories."""

from __future__ import annotations

from importlib import import_module
from threading import Lock
from typing import Callable

from .auto_config import (
    factories_registered,
    register_setup_factories,
)

_LOCK = Lock()


def _lazy_factory(module_path: str, class_name: str) -> Callable[[], object]:
    def _factory() -> object:
        module = import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    return _factory


def ensure_setup_factories_registered() -> None:
    """Register default setup factories if none have been provided."""

    if factories_registered():
        return

    with _LOCK:
        if factories_registered():
            return
        register_setup_factories(
            aws_factory=_lazy_factory("softmax.cli.components.aws", "AWSSetup"),
            observatory_factory=_lazy_factory("softmax.cli.components.observatory_key", "ObservatoryKeySetup"),
            wandb_factory=_lazy_factory("softmax.cli.components.wandb", "WandbSetup"),
        )


__all__ = ["ensure_setup_factories_registered"]
