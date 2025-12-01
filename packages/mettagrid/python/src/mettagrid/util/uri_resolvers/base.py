from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    run_name: str
    epoch: int
    uri: str


@dataclass(frozen=True, slots=True)
class ParsedScheme:
    raw: str
    scheme: str
    canonical: str
    local_path: Optional[Path] = None
    bucket: Optional[str] = None
    key: Optional[str] = None
    path: Optional[str] = None

    def require_local_path(self) -> Path:
        if self.scheme != "file" or self.local_path is None:
            raise ValueError(f"URI '{self.raw}' does not refer to a local file path")
        return self.local_path

    def require_s3(self) -> tuple[str, str]:
        if self.scheme != "s3" or not self.bucket or not self.key:
            raise ValueError(f"URI '{self.raw}' is not an s3:// path")
        return self.bucket, self.key

    @property
    def checkpoint_info(self) -> tuple[str, int] | None:
        if self.scheme == "mock" and self.path:
            return (self.path, 0)
        if not self.path:
            return None
        return _extract_run_and_epoch(Path(self.path))


class SchemeResolver(ABC):
    @property
    @abstractmethod
    def scheme(self) -> str:
        pass

    def matches_scheme(self, uri: str) -> bool:
        return uri.startswith(f"{self.scheme}://")

    @abstractmethod
    def parse(self, uri: str) -> ParsedScheme:
        pass

    def resolve(self, uri: str) -> str:
        return self.parse(uri).canonical
