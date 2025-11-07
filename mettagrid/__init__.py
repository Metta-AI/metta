from __future__ import annotations

import importlib
import pathlib
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_extra = pathlib.Path(__file__).resolve().parents[1] / "packages" / "mettagrid" / "python" / "src" / "mettagrid"
if _extra.is_dir():
    extra_path = str(_extra)
    if extra_path not in __path__:
        __path__.append(extra_path)

__all__ = ["MettaGridConfig"]


def __getattr__(name: str):
    if name == "MettaGridConfig":
        module = importlib.import_module("mettagrid.config.mettagrid_config")
        value = module.MettaGridConfig
        globals()[name] = value
        return value

    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - lazy loader
        raise AttributeError(f"module 'mettagrid' has no attribute '{name}'") from exc

    globals()[name] = module
    return module
