"""Shared URI parsing utilities for Metta.

Provides canonical parsing for supported schemes (local files, file://, s3://,
mock://, Google Drive, HTTP URLs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
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
            if "/" not in remainder:
                raise ValueError("Malformed S3 URI. Expected s3://bucket/key")
            bucket, key = remainder.split("/", 1)
            if not bucket or not key:
                raise ValueError("Malformed S3 URI. Bucket and key must be non-empty")
            return cls(raw=value, scheme="s3", bucket=bucket, key=key, path=key)

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


__all__ = ["WandbURI", "ParsedURI"]
