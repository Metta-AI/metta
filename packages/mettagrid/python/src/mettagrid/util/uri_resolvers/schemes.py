from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, overload
from urllib.parse import unquote, urlparse

import boto3

from mettagrid.util.module import load_symbol
from mettagrid.util.uri_resolvers.base import (
    CheckpointMetadata,
    FileParsedScheme,
    MockParsedScheme,
    ParsedScheme,
    S3ParsedScheme,
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
        return uri.startswith("file://")

    def parse(self, uri: str) -> FileParsedScheme:
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
        return FileParsedScheme(canonical=canonical, local_path=local_path)

    def _get_latest_checkpoint_uri(self, parsed: FileParsedScheme) -> str | None:
        if not parsed.local_path.is_dir():
            return None
        best: tuple[int, str] | None = None
        for ckpt in parsed.local_path.glob("*.mpt"):
            uri = f"file://{ckpt}"
            info = self.parse(uri).checkpoint_info
            if info and (best is None or info[1] > best[0]):
                best = (info[1], uri)
        return best[1] if best else None

    def get_path_to_policy_spec_or_mpt(self, uri: str) -> str:
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            if base_uri.endswith("/"):
                base_uri = base_uri[:-1]
            parsed = self.parse(base_uri)
            latest = self._get_latest_checkpoint_uri(parsed)
            if latest:
                return latest
            raise ValueError(f"No latest checkpoint found for {base_uri}")

        parsed = self.parse(uri)
        if not uri.endswith(".mpt"):
            latest = self._get_latest_checkpoint_uri(parsed)
            if latest:
                return latest

        return parsed.canonical


class S3SchemeResolver(SchemeResolver):
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

    def parse(self, uri: str) -> S3ParsedScheme:
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        remainder = uri[5:]
        if "/" not in remainder:
            raise ValueError("Malformed S3 URI. Expected s3://bucket/key")
        bucket, key = remainder.split("/", 1)
        if not bucket or not key:
            raise ValueError("Malformed S3 URI. Bucket and key must be non-empty")
        canonical = f"s3://{bucket}/{key}"
        return S3ParsedScheme(canonical=canonical, bucket=bucket, key=key)

    def _get_latest_checkpoint_uri(self, parsed: S3ParsedScheme) -> str | None:
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

    def get_path_to_policy_spec_or_mpt(self, uri: str) -> str:
        if uri.endswith(":latest"):
            base_uri = uri[:-7]
            if base_uri.endswith("/"):
                base_uri = base_uri[:-1]
            parsed = self.parse(base_uri)
            latest = self._get_latest_checkpoint_uri(parsed)
            if latest:
                return latest
            raise ValueError(f"No latest checkpoint found for {base_uri}")

        parsed = self.parse(uri)
        if not uri.endswith(".mpt"):
            latest = self._get_latest_checkpoint_uri(parsed)
            if latest:
                return latest

        return parsed.canonical


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

    def parse(self, uri: str) -> S3ParsedScheme:
        if uri.startswith("https://") or uri.startswith("http://"):
            # Match patterns:
            # - https://{bucket}.s3.amazonaws.com/{key}
            # - https://{bucket}.s3.{region}.amazonaws.com/{key}
            s3_pattern = r"^https?://([^.]+)\.s3(?:\.([^.]+))?\.amazonaws\.com/(.+)$"
            match = re.match(s3_pattern, uri)
            if match:
                bucket, _region, key = match.groups()
                return S3ParsedScheme(canonical=f"s3://{bucket}/{key}", bucket=bucket, key=key)
        raise ValueError(f"Expected https:// or http:// URI, got: {uri}")


class MockSchemeResolver(SchemeResolver):
    """Resolves mock URIs for testing.

    Supported formats:
      - mock://policy_name
    """

    @property
    def scheme(self) -> str:
        return "mock"

    def parse(self, uri: str) -> MockParsedScheme:
        if not uri.startswith("mock://"):
            raise ValueError(f"Expected mock:// URI, got: {uri}")
        path = uri[len("mock://") :]
        if not path:
            raise ValueError("mock:// URIs must include a path")
        canonical = f"mock://{path}"
        return MockParsedScheme(canonical=canonical, path=path)


_SCHEME_RESOLVERS: list[str] = [
    "mettagrid.util.uri_resolvers.schemes.FileSchemeResolver",
    "mettagrid.util.uri_resolvers.schemes.S3SchemeResolver",
    "mettagrid.util.uri_resolvers.schemes.HttpSchemeResolver",
    "mettagrid.util.uri_resolvers.schemes.HttpSchemeResolver",
    "mettagrid.util.uri_resolvers.schemes.MockSchemeResolver",
    "metta.rl.metta_scheme_resolver.MettaSchemeResolver",
]


def _get_resolver(uri: str, default_scheme: str | None = "file") -> SchemeResolver | None:
    if not uri:
        raise ValueError("URI cannot be empty")

    if default_scheme and "://" not in uri:
        uri = f"{default_scheme}://{uri}"

    for resolver_path in _SCHEME_RESOLVERS:
        resolver_class: type[SchemeResolver] | None = load_symbol(resolver_path, strict=False)  # type: ignore[assignment]
        if resolver_class and resolver_class().matches_scheme(uri):
            return resolver_class()
    return None


@overload
def parse_uri(uri: str, allow_none: Literal[False], **kwargs) -> ParsedScheme: ...
@overload
def parse_uri(uri: str, allow_none: Literal[True], **kwargs) -> ParsedScheme | None: ...
@overload
def parse_uri(uri: str, allow_none: bool, **kwargs) -> ParsedScheme | None: ...


def parse_uri(uri: str, allow_none: bool = False, **kwargs) -> ParsedScheme | None:
    resolver = _get_resolver(uri, **kwargs)
    if resolver is None:
        if allow_none:
            return None
        raise ValueError("Invalid URI")
    return resolver.parse(uri)


def resolve_uri(uri: str) -> ParsedScheme:
    """Resolve a URI to its canonical form, finding :latest checkpoints if needed."""
    resolver = _get_resolver(uri)
    if not resolver:
        raise ValueError("Unsupported URI")
    resolved_uri_str = resolver.get_path_to_policy_spec_or_mpt(uri)
    return parse_uri(resolved_uri_str, allow_none=False)


def checkpoint_filename(run_name: str, epoch: int) -> str:
    return f"{run_name}:v{epoch}.mpt"


def get_checkpoint_metadata(uri: str) -> CheckpointMetadata:
    parsed = resolve_uri(uri)
    info = parsed.checkpoint_info
    if not info:
        raise ValueError(f"Could not extract checkpoint metadata from {uri}")
    return CheckpointMetadata(run_name=info[0], epoch=info[1], uri=parsed.canonical)


def policy_spec_from_uri(
    uri: str, *, device: str = "cpu", strict: bool = True, remove_downloaded_copy_on_exit: bool = False
):
    from mettagrid.policy.policy import PolicySpec
    from mettagrid.policy.prepare_policy_spec import (
        download_policy_spec_from_s3_as_zip,
        load_policy_spec_from_local_dir,
        load_policy_spec_from_zip,
    )

    parsed = resolve_uri(uri)

    if parsed.canonical.endswith(".mpt"):
        checkpoint_path = str(parsed.local_path) if parsed.local_path else parsed.canonical
        return PolicySpec(
            class_path="mettagrid.policy.mpt_policy.MptPolicy",
            init_kwargs={
                "checkpoint_uri": checkpoint_path,
                "device": device,
                "strict": strict,
                "allow_legacy_architecture": parsed.scheme == "s3",
            },
        )

    if parsed.scheme == "s3":
        local = download_policy_spec_from_s3_as_zip(
            s3_path=parsed.canonical,
            remove_downloaded_copy_on_exit=remove_downloaded_copy_on_exit,
        )
        parsed = resolve_uri(local.as_uri())

    if parsed.local_path:
        if parsed.local_path.is_file():
            return load_policy_spec_from_zip(
                parsed.local_path,
                device=device,
                remove_downloaded_copy_on_exit=remove_downloaded_copy_on_exit,
            )
        return load_policy_spec_from_local_dir(parsed.local_path, device=device)

    raise ValueError(f"Cannot load policy spec from URI: {uri}")
