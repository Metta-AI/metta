from __future__ import annotations

import importlib
import pathlib
import pkgutil
import typing

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_extra = pathlib.Path(__file__).resolve().parents[3] / "common" / "src" / "metta" / "common" / "tool"
if _extra.is_dir():
    extra_path = str(_extra)
    if extra_path not in __path__:
        __path__.append(extra_path)

try:
    tool_module = importlib.import_module("metta.common.tool.tool")
except ImportError:  # pragma: no cover - optional dependency
    Tool: typing.Any | None = None
else:
    Tool = tool_module.Tool

__all__ = ["Tool"]
