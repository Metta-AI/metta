from __future__ import annotations

import pathlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_extra = pathlib.Path(__file__).resolve().parents[1] / "packages" / "mettagrid" / "python" / "src" / "mettagrid"
if _extra.is_dir():
    extra_path = str(_extra)
    if extra_path not in __path__:
        __path__.append(extra_path)

try:
    from .config.mettagrid_config import MettaGridConfig  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional during bootstrap
    MettaGridConfig = None  # type: ignore[assignment]

__all__: list[str] = ["MettaGridConfig"]
