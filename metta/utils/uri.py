"""Shared URI parsing utilities for Metta.

Provides canonical parsing for supported schemes (local files, file://, mock://, HTTP URLs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse


@dataclass(frozen=True, slots=True)
class ParsedURI:
    """Canonical representation for supported URI schemes."""

    raw: str
    scheme: str
    local_path: Optional[Path] = None
    bucket: Optional[str] = None
    key: Optional[str] = None
    path: Optional[str] = None

    @property
    def canonical(self) -> str:
        """Return a normalized string representation."""
        if self.scheme == "file" and self.local_path is not None:
            return self.local_path.as_uri()
        if self.scheme == "s3" and self.bucket and self.key:
            return f"s3://{self.bucket}/{self.key}"
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

    def is_remote(self) -> bool:
        """Return True if the URI references a remote resource."""
        return self.scheme in {"s3", "gdrive", "http"}

    @classmethod
    def parse(cls, value: str) -> "ParsedURI":
        if not value:
            raise ValueError("URI cannot be empty")

        # Check if this is an S3 HTTPS URL and convert to s3:// URI
        if value.startswith("https://") or value.startswith("http://"):
            # Match patterns: 
            # - https://{bucket}.s3.amazonaws.com/{key}
            # - https://{bucket}.s3.{region}.amazonaws.com/{key}
            s3_pattern = r"^https?://([^.]+)\.s3(?:\.([^.]+))?\.amazonaws\.com/(.+)$"
            match = re.match(s3_pattern, value)
            if match:
                bucket, key = match.groups()
                # Convert to s3:// URI for proper handling
                value = f"s3://{bucket}/{key}"

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


__all__ = ["ParsedURI"]
