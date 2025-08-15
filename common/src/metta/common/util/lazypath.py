import os
from pathlib import Path


class LazyPath:
    def __init__(self, path_str):
        self._path = Path(os.path.expanduser(path_str))

    def __str__(self):
        self._ensure_exists()
        return str(self._path)

    def __fspath__(self):
        self._ensure_exists()
        return os.fspath(self._path)

    def __repr__(self):
        return f"LazyPath({str(self)})"

    def __getattr__(self, name):
        self._ensure_exists()
        return getattr(self._path, name)

    def __call__(self) -> Path:
        return self.get()

    def _ensure_exists(self):
        self._path.touch(exist_ok=True)

    def get(self) -> Path:
        self._ensure_exists()
        return self._path
