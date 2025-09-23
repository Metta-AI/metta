"""Shared URI parsing utilities for Metta.

Provides canonical parsing for supported schemes (local files, file://, s3://,
mock://, Google Drive, HTTP URLs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse


@dataclass(frozen=True, slots=True)
class WandbURI:
    """Parsed representation of a W&B artifact URI."""

    entity: str
    project: str
    artifact_path: str
    version: str = "latest"

    @classmethod
    def parse(cls, uri: str) -> "WandbURI":
        if not uri.startswith("wandb://"):
            raise ValueError("W&B URI must start with wandb://")

        body = uri[len("wandb://") :]
        if ":" in body:
            path_part, version = body.rsplit(":", 1)
        else:
            path_part, version = body, "latest"

        if "/" not in path_part:
            raise ValueError(
                "Malformed W&B URI. Expected fully-qualified form: wandb://entity/project/artifact_path:version"
            )

        parts = path_part.split("/")
        if len(parts) < 3:
            raise ValueError(
                "Malformed W&B URI. Expected `wandb://entity/project/artifact_path:version`. "
                "Example: wandb://my-entity/metta/model/run-name:latest"
            )

        entity = parts[0]
        project = parts[1]
        artifact_path = "/".join(parts[2:])
        if not project or not artifact_path:
            raise ValueError("Project and artifact path must be non-empty")

        return cls(entity, project, artifact_path, version)

    def qname(self) -> str:
        """Qualified name accepted by `wandb.Api().artifact(...)`."""
        return f"{self.entity}/{self.project}/{self.artifact_path}:{self.version}"

    def http_url(self) -> str:
        """Human-readable URL for this artifact version."""
        return f"https://wandb.ai/{self.entity}/{self.project}/artifacts/{self.artifact_path}/{self.version}"

    def __str__(self) -> str:  # pragma: no cover - repr helper
        return f"wandb://{self.entity}/{self.project}/{self.artifact_path}:{self.version}"


@dataclass(frozen=True, slots=True)
class ParsedURI:
    """Canonical representation for supported URI schemes."""

    raw: str
    scheme: str
    local_path: Optional[Path] = None
    bucket: Optional[str] = None
    key: Optional[str] = None
    wandb: Optional[WandbURI] = None
    path: Optional[str] = None

    @property
    def canonical(self) -> str:
        """Return a normalized string representation."""
        if self.scheme == "file" and self.local_path is not None:
            return self.local_path.as_uri()
        if self.scheme == "s3" and self.bucket and self.key:
            return f"s3://{self.bucket}/{self.key}"
        if self.scheme == "wandb" and self.wandb is not None:
            return str(self.wandb)
        if self.scheme == "mock":
            return f"mock://{self.path or ''}"
        return self.raw

    def require_local_path(self) -> Path:
        if self.scheme != "file" or self.local_path is None:
            raise ValueError(f"URI '{self.raw}' does not refer to a local file path")
        return self.local_path

    def require_s3(self) -> tuple[str, str]:
        if self.scheme != "s3" or not self.bucket or not self.key:
            raise ValueError(f"URI '{self.raw}' is not an s3:// path")
        return self.bucket, self.key

    def require_wandb(self) -> WandbURI:
        if self.scheme != "wandb" or self.wandb is None:
            raise ValueError(f"URI '{self.raw}' is not a wandb:// artifact")
        return self.wandb

    def is_remote(self) -> bool:
        """Return True if the URI references a remote resource."""
        return self.scheme in {"s3", "wandb", "gdrive", "http"}

    @classmethod
    def parse(cls, value: str) -> "ParsedURI":
        if not value:
            raise ValueError("URI cannot be empty")

        if value.startswith("wandb://"):
            wandb_uri = WandbURI.parse(value)
            return cls(
                raw=value,
                scheme="wandb",
                wandb=wandb_uri,
                path=wandb_uri.artifact_path,
            )

        if value.startswith("s3://"):
            remainder = value[5:]
            if not remainder:
                raise ValueError("Malformed S3 URI. Expected s3://bucket[/key]")
            if "/" in remainder:
                bucket, key = remainder.split("/", 1)
                key = key or None
            else:
                bucket, key = remainder, None
            if not bucket:
                raise ValueError("Malformed S3 URI. Bucket must be non-empty")
            path = key if key else None
            return cls(raw=value, scheme="s3", bucket=bucket, key=key, path=path)

        if value.startswith("mock://"):
            path = value[len("mock://") :]
            if not path:
                raise ValueError("mock:// URIs must include a path")
            return cls(raw=value, scheme="mock", path=path)

        if value.startswith("gdrive://") or value.startswith("https://drive.google.com/"):
            return cls(raw=value, scheme="gdrive", path=value)

        if value.startswith("file://"):
            parsed = urlparse(value)
            # Combine netloc + path to support file://localhost/tmp
            combined_path = unquote(parsed.path)
            if parsed.netloc:
                combined_path = f"{parsed.netloc}{combined_path}"
            if not combined_path:
                raise ValueError(f"Malformed file URI: {value}")
            local_path = Path(combined_path).expanduser().resolve()
            return cls(raw=value, scheme="file", local_path=local_path, path=str(local_path))

        if value.startswith("http://") or value.startswith("https://"):
            return cls(raw=value, scheme="http", path=value)

        # Treat everything else as a local filesystem path
        local_path = Path(value).expanduser().resolve()
        return cls(raw=value, scheme="file", local_path=local_path, path=str(local_path))


def _clean_segments(segments: Iterable[str]) -> list[str]:
    return [segment.strip("/") for segment in segments if segment and segment.strip("/")]


def artifact_join(base: str | Path | None, *segments: str) -> Optional[str]:
    if base is None:
        return None

    base_str = str(base).strip()
    cleaned = _clean_segments(segments)

    if not cleaned:
        return base_str

    if base_str.startswith("s3://"):
        remainder = base_str[5:]
        if not remainder:
            raise ValueError("S3 URI must include a bucket name")
        bucket, sep, key_prefix = remainder.partition("/")
        key_parts: list[str] = []
        if sep and key_prefix:
            key_parts.append(key_prefix.rstrip("/"))
        key_parts.extend(cleaned)
        joined = "/".join(part for part in key_parts if part)
        return f"s3://{bucket}/{joined}" if joined else f"s3://{bucket}"

    if base_str.startswith("gdrive://"):
        prefix = base_str[len("gdrive://") :].rstrip("/")
        joined = "/".join(filter(None, [prefix, *cleaned]))
        return f"gdrive://{joined}" if joined else base_str

    if base_str.startswith(("http://", "https://")):
        return f"{base_str.rstrip('/')}/{'/'.join(cleaned)}"

    return str(Path(base_str).joinpath(*cleaned))


def artifact_policy_run_root(
    base: str | Path | None,
    *,
    run_name: Optional[str],
    epoch: Optional[int],
) -> Optional[str]:
    if base is None or not run_name:
        return artifact_join(base)
    root = artifact_join(base, run_name)
    if epoch:
        root = artifact_join(root, f"v{epoch}")
    return root


def artifact_simulation_root(
    base: str | Path | None,
    *,
    suite: str,
    name: str,
    simulation_id: Optional[str] = None,
) -> Optional[str]:
    root = artifact_join(base, suite, name)
    if simulation_id:
        root = artifact_join(root, simulation_id)
    return root


__all__ = [
    "WandbURI",
    "ParsedURI",
    "artifact_join",
    "artifact_policy_run_root",
    "artifact_simulation_root",
]
