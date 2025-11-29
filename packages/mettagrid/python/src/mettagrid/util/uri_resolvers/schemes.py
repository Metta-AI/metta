from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3

from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.base import (
    CheckpointMetadata,
    ParsedScheme,
    SchemeResolver,
)


class FileSchemeResolver(SchemeResolver):
    """Resolves local filesystem URIs.

    Supported formats:
      - file:///absolute/path/to/file.mpt
      - file://relative/path/to/file.mpt
      - /absolute/path/to/file.mpt  (bare path, no scheme)
      - relative/path/to/file.mpt   (bare path, no scheme)
      - ~/path/to/file.mpt          (expands ~)
      - /path/to/checkpoints:latest (resolves to highest epoch .mpt in dir)
    """

    @property
    def scheme(self) -> str:
        return "file"

    def matches_scheme(self, uri: str) -> bool:
        return uri.startswith("file://") or "://" not in uri

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

    def _can_find_latest(self, parsed: ParsedScheme) -> bool:
        return parsed.local_path is not None

    def _get_latest_checkpoint_uri(self, parsed: ParsedScheme) -> str | None:
        if not parsed.local_path or not parsed.local_path.is_dir():
            return None
        best: tuple[int, str] | None = None
        for ckpt in parsed.local_path.glob("*.mpt"):
            uri = f"file://{ckpt}"
            info = self.parse(uri).checkpoint_info
            if info and (best is None or info[1] > best[0]):
                best = (info[1], uri)
        return best[1] if best else None

    def resolve(self, uri: str) -> str:
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            if base_uri.endswith("/"):
                base_uri = base_uri[:-1]
            parsed = self.parse(base_uri)
            if self._can_find_latest(parsed):
                latest = self._get_latest_checkpoint_uri(parsed)
                if latest:
                    return latest
            raise ValueError(f"No latest checkpoint found for {base_uri}")

        parsed = self.parse(uri)

        if self._can_find_latest(parsed) and not uri.endswith(".mpt"):
            latest = self._get_latest_checkpoint_uri(parsed)
            if latest:
                return latest

        return parsed.canonical


class S3SchemeResolver(FileSchemeResolver):
    """Resolves AWS S3 URIs.

    Supported formats:
      - s3://bucket/path/to/file.mpt
      - s3://bucket/path/to/checkpoints:latest (resolves to highest epoch .mpt)
    """

    @property
    def scheme(self) -> str:
        return "s3"

    def matches_scheme(self, uri: str) -> bool:
        return uri.startswith("s3://")

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

    def _can_find_latest(self, parsed: ParsedScheme) -> bool:
        return parsed.bucket is not None and parsed.key is not None

    def _get_latest_checkpoint_uri(self, parsed: ParsedScheme) -> str | None:
        if not parsed.bucket or not parsed.key:
            return None
        prefix = parsed.key
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)
        if response["KeyCount"] == 0:
            return None
        best: tuple[int, str] | None = None
        for obj in response["Contents"]:
            if not obj["Key"].endswith(".mpt"):
                continue
            uri = f"s3://{parsed.bucket}/{obj['Key']}"
            info = self.parse(uri).checkpoint_info
            if info and (best is None or info[1] > best[0]):
                best = (info[1], uri)
        return best[1] if best else None


class HttpSchemeResolver(SchemeResolver):
    """Resolves HTTP/HTTPS S3 URLs to s3:// URIs.

    Supported formats:
      - https://{bucket}.s3.amazonaws.com/{key}
      - https://{bucket}.s3.{region}.amazonaws.com/{key}
      - http://{bucket}.s3.amazonaws.com/{key}
    """

    @property
    def scheme(self) -> str:
        return "https"

    def matches_scheme(self, uri: str) -> bool:
        return uri.startswith("https://") or uri.startswith("http://")

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
    """Resolves mock URIs for testing.

    Supported formats:
      - mock://policy_name
    """

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
    "file": "mettagrid.util.uri_resolvers.schemes.FileSchemeResolver",
    "s3": "mettagrid.util.uri_resolvers.schemes.S3SchemeResolver",
    "http": "mettagrid.util.uri_resolvers.schemes.HttpSchemeResolver",
    "https": "mettagrid.util.uri_resolvers.schemes.HttpSchemeResolver",
    "mock": "mettagrid.util.uri_resolvers.schemes.MockSchemeResolver",
    "metta": "metta.rl.metta_scheme_resolver.MettaSchemeResolver",
}


def _get_resolver(uri: str) -> SchemeResolver:
    if not uri:
        raise ValueError("URI cannot be empty")

    for resolver_path in SCHEME_RESOLVERS.values():
        resolver_class: type[SchemeResolver] | None = load_symbol(resolver_path, strict=False)  # type: ignore[assignment]
        if resolver_class and resolver_class().matches_scheme(uri):
            return resolver_class()

    scheme = uri.split("://", 1)[0] if "://" in uri else None
    raise ValueError(f"Unsupported URI scheme: {scheme}://" if scheme else f"No resolver found for URI: {uri}")


def parse_uri(uri: str) -> ParsedScheme:
    return _get_resolver(uri).parse(uri)


def resolve_uri(uri: str) -> str:
    return _get_resolver(uri).resolve(uri)


def checkpoint_filename(run_name: str, epoch: int) -> str:
    return f"{run_name}:v{epoch}.mpt"


def get_checkpoint_metadata(uri: str) -> CheckpointMetadata:
    resolved = resolve_uri(uri)
    parsed = parse_uri(resolved)
    info = parsed.checkpoint_info
    if not info:
        raise ValueError(f"Could not extract checkpoint metadata from {uri}")
    return CheckpointMetadata(run_name=info[0], epoch=info[1], uri=resolved)


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
