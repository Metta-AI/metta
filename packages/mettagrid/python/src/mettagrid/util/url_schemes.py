from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import unquote, urlparse

import boto3

from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from typing import Any


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


class SchemeResolver(ABC):
    @property
    @abstractmethod
    def scheme(self) -> str:
        pass

    @abstractmethod
    def parse(self, uri: str) -> ParsedScheme:
        pass

    def resolve(self, uri: str) -> str:
        return self.parse(uri).canonical


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


def _find_latest_checkpoint(checkpoints: list[Path]) -> Path | None:
    best: tuple[int, Path] | None = None
    for ckpt in checkpoints:
        meta = _extract_run_and_epoch(ckpt)
        if meta and (best is None or meta[1] > best[0]):
            best = (meta[1], ckpt)
    return best[1] if best else None


class FileSchemeResolver(SchemeResolver):
    @property
    def scheme(self) -> str:
        return "file"

    def parse(self, uri: str) -> ParsedScheme:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            combined_path = unquote(parsed.path)
            if parsed.netloc:
                combined_path = f"{parsed.netloc}{combined_path}"
            if not combined_path:
                raise ValueError(f"Malformed file URI: {uri}")
            local_path = Path(combined_path).expanduser().resolve()
        else:
            local_path = Path(uri).expanduser().resolve()

        canonical = local_path.as_uri()
        return ParsedScheme(
            raw=uri, scheme=self.scheme, canonical=canonical, local_path=local_path, path=str(local_path)
        )

    def _get_latest_checkpoint_uri(self, local_path: Path) -> str | None:
        if not local_path.is_dir():
            return None
        checkpoints = [ckpt for ckpt in local_path.glob("*.mpt") if ckpt.stem]
        latest = _find_latest_checkpoint(checkpoints)
        return f"file://{latest}" if latest else None

    def resolve(self, uri: str) -> str:
        # Handle /:latest suffix
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            if base_uri.endswith("/"):
                base_uri = base_uri[:-1]
            parsed = self.parse(base_uri)
            if parsed.local_path:
                latest = self._get_latest_checkpoint_uri(parsed.local_path)
                if latest:
                    return latest
            raise ValueError(f"No latest checkpoint found for {base_uri}")

        parsed = self.parse(uri)

        # If not pointing to .mpt file, try to find latest
        if parsed.local_path and not uri.endswith(".mpt"):
            latest = self._get_latest_checkpoint_uri(parsed.local_path)
            if latest:
                return latest

        return parsed.canonical


class S3SchemeResolver(SchemeResolver):
    @property
    def scheme(self) -> str:
        return "s3"

    def parse(self, uri: str) -> ParsedScheme:
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        remainder = uri[5:]
        if "/" not in remainder:
            raise ValueError("Malformed S3 URI. Expected s3://bucket/key")
        bucket, key = remainder.split("/", 1)
        if not bucket or not key:
            raise ValueError("Malformed S3 URI. Bucket and key must be non-empty")
        canonical = f"s3://{bucket}/{key}"
        return ParsedScheme(raw=uri, scheme=self.scheme, canonical=canonical, bucket=bucket, key=key, path=key)

    def _get_latest_checkpoint_uri(self, bucket: str, prefix: str) -> str | None:
        # Ensure prefix ends with / for directory-like semantics
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if response["KeyCount"] == 0:
            return None
        checkpoints = [Path(obj["Key"]) for obj in response["Contents"] if obj["Key"].endswith(".mpt")]
        latest = _find_latest_checkpoint(checkpoints)
        return f"s3://{bucket}/{latest}" if latest else None

    def resolve(self, uri: str) -> str:
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            if base_uri.endswith("/"):
                base_uri = base_uri[:-1]
            parsed = self.parse(base_uri)
            if parsed.bucket and parsed.key:
                latest = self._get_latest_checkpoint_uri(parsed.bucket, parsed.key)
                if latest:
                    return latest
            raise ValueError(f"No latest checkpoint found for {base_uri}")

        parsed = self.parse(uri)

        # If not pointing to .mpt file, try to find latest
        if parsed.bucket and parsed.key and not uri.endswith(".mpt"):
            latest = self._get_latest_checkpoint_uri(parsed.bucket, parsed.key)
            if latest:
                return latest

        return parsed.canonical


class HttpSchemeResolver(SchemeResolver):
    @property
    def scheme(self) -> str:
        return "https"

    def parse(self, uri: str) -> ParsedScheme:
        if uri.startswith("https://") or uri.startswith("http://"):
            # Match patterns:
            # - https://{bucket}.s3.amazonaws.com/{key}
            # - https://{bucket}.s3.{region}.amazonaws.com/{key}
            s3_pattern = r"^https?://([^.]+)\.s3(?:\.([^.]+))?\.amazonaws\.com/(.+)$"
            match = re.match(s3_pattern, uri)
            if match:
                bucket, region, key = match.groups()
                # region is optional (None if not present), but we don't need it for s3:// URIs
                # Convert to s3:// URI for proper handling
                value = f"s3://{bucket}/{key}"
                return S3SchemeResolver().parse(value)
        raise ValueError(f"Expected https:// or http:// URI, got: {uri}")


class MockSchemeResolver(SchemeResolver):
    @property
    def scheme(self) -> str:
        return "mock"

    def parse(self, uri: str) -> ParsedScheme:
        if not uri.startswith("mock://"):
            raise ValueError(f"Expected mock:// URI, got: {uri}")
        path = uri[len("mock://") :]
        if not path:
            raise ValueError("mock:// URIs must include a path")
        canonical = f"mock://{path}"
        return ParsedScheme(raw=uri, scheme=self.scheme, canonical=canonical, path=path)


SCHEME_RESOLVERS: dict[str, str] = {
    "file": "mettagrid.util.url_schemes.FileSchemeResolver",
    "s3": "mettagrid.util.url_schemes.S3SchemeResolver",
    "https": "mettagrid.util.url_schemes.HttpSchemeResolver",
    "mock": "mettagrid.util.url_schemes.MockSchemeResolver",
    "metta": "metta.rl.metta_scheme_resolver.MettaSchemeResolver",
}


def get_scheme_resolver(scheme: str) -> SchemeResolver | None:
    resolver_path = SCHEME_RESOLVERS.get(scheme)
    if not resolver_path:
        return None
    resolver_class: Any = load_symbol(resolver_path, strict=False)
    if resolver_class is None:
        return None
    return resolver_class()


def _get_scheme_and_resolver(uri: str) -> tuple[str | None, SchemeResolver]:
    if not uri:
        raise ValueError("URI cannot be empty")

    scheme = None
    if "://" in uri:
        scheme = uri.split("://", 1)[0]

    if scheme:
        resolver = get_scheme_resolver(scheme)
        if resolver is None:
            raise ValueError(f"Unsupported URI scheme: {scheme}://")
        return scheme, resolver

    return None, FileSchemeResolver()


def parse_uri(uri: str) -> ParsedScheme:
    _, resolver = _get_scheme_and_resolver(uri)
    return resolver.parse(uri)


def resolve_uri(uri: str) -> str:
    _, resolver = _get_scheme_and_resolver(uri)
    return resolver.resolve(uri)


def checkpoint_filename(run_name: str, epoch: int) -> str:
    return f"{run_name}:v{epoch}.mpt"


class CheckpointMetadata(dict):
    run_name: str
    epoch: int
    uri: str


def key_and_version(uri: str) -> tuple[str, int] | None:
    parsed = parse_uri(uri)
    if parsed.scheme == "mock" and parsed.path:
        return (parsed.path, 0)
    if parsed.scheme == "file" and parsed.local_path:
        file_path = Path(parsed.local_path)
    elif parsed.scheme == "s3" and parsed.key:
        file_path = Path(parsed.key)
    else:
        raise ValueError(f"Could not extract key and version from {uri}")
    return _extract_run_and_epoch(file_path)


def get_checkpoint_metadata(uri: str) -> CheckpointMetadata:
    normalized_uri = resolve_uri(uri)
    metadata = key_and_version(normalized_uri)
    if not metadata:
        raise ValueError(f"Could not extract metadata from uri {uri}")
    run_name, epoch = metadata
    return CheckpointMetadata(run_name=run_name, epoch=epoch, uri=normalized_uri)


def policy_spec_from_uri(uri: str, *, device: str = "cpu", strict: bool = True):
    from mettagrid.policy.policy import PolicySpec

    normalized_uri = resolve_uri(uri)
    return PolicySpec(
        class_path="mettagrid.policy.mpt_policy.MptPolicy",
        init_kwargs={
            "checkpoint_uri": normalized_uri,
            "device": device,
            "strict": strict,
        },
    )
