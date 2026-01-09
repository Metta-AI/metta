from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel


def _extract_run_and_epoch(path_str: str) -> tuple[str, int] | None:
    stem = Path(path_str).stem
    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


class CheckpointMetadata(BaseModel, frozen=True):
    run_name: str
    epoch: int
    uri: str


class FileParsedScheme(BaseModel, frozen=True):
    scheme: Literal["file"] = "file"
    canonical: str
    local_path: Path

    @property
    def checkpoint_info(self) -> tuple[str, int] | None:
        return _extract_run_and_epoch(str(self.local_path))


class S3ParsedScheme(BaseModel, frozen=True):
    scheme: Literal["s3"] = "s3"
    canonical: str
    bucket: str
    key: str

    @property
    def local_path(self) -> None:
        return None

    @property
    def checkpoint_info(self) -> tuple[str, int] | None:
        return _extract_run_and_epoch(self.key)


class MockParsedScheme(BaseModel, frozen=True):
    scheme: Literal["mock"] = "mock"
    canonical: str
    path: str

    @property
    def local_path(self) -> None:
        return None

    @property
    def checkpoint_info(self) -> tuple[str, int] | None:
        return (self.path, 0)


class MettaParsedScheme(BaseModel, frozen=True):
    scheme: Literal["metta"] = "metta"
    canonical: str
    path: str

    @property
    def local_path(self) -> None:
        return None

    @property
    def checkpoint_info(self) -> tuple[str, int] | None:
        return _extract_run_and_epoch(self.path)


ParsedScheme = Union[FileParsedScheme, S3ParsedScheme, MockParsedScheme, MettaParsedScheme]


class SchemeResolver(ABC):
    @property
    @abstractmethod
    def scheme(self) -> str:
        pass

    @property
    def _expected_prefix(self) -> str:
        return f"{self.scheme}://"

    def matches_scheme(self, uri: str) -> bool:
        return uri.startswith(self._expected_prefix)

    @abstractmethod
    def parse(self, uri: str) -> ParsedScheme:
        pass

    def get_path_to_policy_spec(self, uri: str) -> str:
        return self.parse(uri).canonical
