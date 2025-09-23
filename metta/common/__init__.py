"""Namespace package bridging workspace-provided common utilities."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_repo_root = Path(__file__).resolve().parents[2]
_workspace_common = _repo_root / "common" / "src" / "metta" / "common"
if _workspace_common.exists():
    _workspace_str = str(_workspace_common)
    if _workspace_str not in __path__:
        __path__.append(_workspace_str)

__all__: list[str] = []
