"""Utilities for managing CUDA-related environment quirks at import time."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any


_PATCHED_TORCH_BACKENDS = False


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except RuntimeError:
        # Some FUSE mounts (e.g., inside Dev Containers) can raise RuntimeError on resolve
        return path


def prune_conflicting_nvidia_paths() -> None:
    """Strip NVIDIA vendor libraries from other virtualenvs off LD_LIBRARY_PATH.

    When `uv sync --no-build-isolation-package pufferlib` runs, PyTorch and the
    NVIDIA wheels install into the project venv. If a user had previously ran a
    different Metta checkout, their shell profile may have appended that other
    venv's `site-packages/nvidia/<pkg>/lib` directories to `LD_LIBRARY_PATH`.

    Mixing vendor libraries compiled for different CUDA minor versions causes
    runtime crashes (e.g., cuDNN 9.10 being picked up while torch expects 9.13).

    This helper keeps only entries that belong to the *current* interpreter's
    prefix, leaving system CUDA paths (e.g., `/usr/local/cuda-13.0/lib64`) as-is.
    """

    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if not ld_path:
        return

    entries = [entry for entry in ld_path.split(":") if entry]
    if not entries:
        return

    prefix = Path(sys.prefix).resolve()
    cleaned: list[str] = []

    for raw_entry in entries:
        entry = Path(raw_entry)
        if "site-packages" in entry.parts and "nvidia" in entry.parts:
            resolved = _resolve(entry)
            try:
                if not resolved.is_relative_to(prefix):
                    # Skip vendor dirs from other environments
                    continue
            except AttributeError:  # Python < 3.9 fallback
                if str(resolved).startswith(str(prefix)) is False:
                    continue
        cleaned.append(raw_entry)

    if cleaned != entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)


def configure_torch_backends(torch_module: Any) -> None:
    """Gracefully degrade torch backends when the bundled cuDNN is missing."""

    global _PATCHED_TORCH_BACKENDS
    if _PATCHED_TORCH_BACKENDS:
        return

    if not torch_module.cuda.is_available():
        _PATCHED_TORCH_BACKENDS = True
        return

    try:
        torch_module.backends.cudnn.version()
    except RuntimeError as err:  # pragma: no cover - depends on host runtime
        warnings.warn(
            f"cuDNN runtime mismatch detected ({err}); disabling cuDNN acceleration for this run.",
            RuntimeWarning,
        )

        def _deny(*_args: Any, **_kwargs: Any) -> bool:
            return False

        torch_module.backends.cudnn.enabled = False
        torch_module.backends.cudnn.is_available = lambda: False  # type: ignore[assignment]
        torch_module.backends.cudnn.is_acceptable = _deny  # type: ignore[assignment]

    _PATCHED_TORCH_BACKENDS = True
