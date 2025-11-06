from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_EXTRA_PATHS = [
    _REPO_ROOT,
    _REPO_ROOT / "agent" / "src",
    _REPO_ROOT / "app_backend" / "src",
    _REPO_ROOT / "common" / "src",
    _REPO_ROOT / "packages" / "mettagrid" / "python" / "src",
    _REPO_ROOT / "packages" / "cogames" / "src",
    _REPO_ROOT / "packages" / "cortex" / "src",
    _REPO_ROOT / "packages" / "gitta" / "src",
    _REPO_ROOT / "packages" / "codebot" / "src",
    _REPO_ROOT / "packages" / "pufferlib-core" / "src",
    _REPO_ROOT / "packages" / "tribal_village" / "src",
    _REPO_ROOT / "softmax" / "src",
    _REPO_ROOT / "gitta" / "src",
    _REPO_ROOT / "external" / "agalite" / "src",
    _REPO_ROOT / "gridworks" / "src",
]

for path in _EXTRA_PATHS:
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
