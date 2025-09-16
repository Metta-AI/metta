"""Local filesystem-backed state store for scheduler-managed experiment state.

Stores JSON files under `<base_dir>/state/<namespace>/<key>.json`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from metta.adaptive.protocols import StateStore


class FileStateStore(StateStore):
    """Simple file-based implementation of StateStore."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def _path(self, namespace: str, key: str) -> str:
        safe_ns = namespace.replace("/", "_")
        safe_key = key.replace("/", "_")
        return os.path.join(self._base_dir, "state", safe_ns, f"{safe_key}.json")

    def get(self, namespace: str, key: str) -> dict | None:
        path = self._path(namespace, key)
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            # Be conservative: on error, treat as missing
            return None

    def put(self, namespace: str, key: str, value: dict) -> None:
        path = self._path(namespace, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure value is JSON serializable; fall back to str for unknown types
        def _default(obj: Any) -> str:
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=_default)
