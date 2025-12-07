from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from mettagrid.util.uri_resolvers.base import ParsedScheme
from mettagrid.util.uri_resolvers.schemes import parse_uri

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedURI:
    """Canonical representation for supported URI schemes."""

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

    @classmethod
    def parse(cls, value: str) -> "ParsedURI":
        parsed: ParsedScheme = parse_uri(value)
        return cls(
            raw=parsed.raw,
            scheme=parsed.scheme,
            canonical=parsed.canonical,
            local_path=parsed.local_path,
            bucket=parsed.bucket,
            key=parsed.key,
            path=parsed.path,
        )


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """Write in-memory bytes/str to *local*, *s3://* destinations."""
    if isinstance(data, str):
        data = data.encode()

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        bucket, key = parsed.require_s3()
        try:
            boto3.client("s3").put_object(Body=data, Bucket=bucket, Key=key, ContentType=content_type)
            logger.debug("Wrote %d B → %s", len(data), http_url(parsed.canonical))
            return
        except NoCredentialsError as e:  # pragma: no cover - environment dependent
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise e

    if parsed.scheme == "file" and parsed.local_path is not None:
        local_path = parsed.local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.debug("Wrote %d B → %s", len(data), local_path)
        return

    raise ValueError(f"Unsupported URI for write_data: {path}")


def exists(path: str) -> bool:
    """Return True if path points to an existing local file or S3 object."""
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        bucket, key = parsed.require_s3()
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "403", "NoSuchKey"}:
                return False
            raise

    if parsed.scheme == "file" and parsed.local_path is not None:
        return parsed.local_path.exists()

    if parsed.scheme == "mock":
        return True

    return False


def read(path: str) -> bytes:
    """Read bytes from a local path or S3 object."""
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        bucket, key = parsed.require_s3()
        try:
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            logger.debug("Read %d B from %s", len(body), parsed.canonical)
            return body
        except NoCredentialsError:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise

    if parsed.scheme == "file" and parsed.local_path is not None:
        data = parsed.local_path.read_bytes()
        logger.debug("Read %d B from %s", len(data), parsed.local_path)
        return data

    raise ValueError(f"Unsupported URI for read(): {path}")


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """Upload a file from disk to s3://, or copy locally."""
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        bucket, key = parsed.require_s3()
        boto3.client("s3").upload_file(local_file, bucket, key, ExtraArgs={"ContentType": content_type})
        logger.debug("Uploaded %s → %s (size %d B)", local_file, parsed.canonical, os.path.getsize(local_file))
        return

    if parsed.scheme == "file" and parsed.local_path is not None:
        dst = parsed.local_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dst)
        logger.debug("Copied %s → %s (size %d B)", local_file, dst, Path(local_file).stat().st_size)
        return

    raise ValueError(f"Unsupported URI for write_file: {path}")


@contextmanager
def local_copy(path: str):
    """Yield a local Path for path (supports local paths and s3:// URIs).

    Local paths are yielded as-is. Remote S3 URIs are downloaded to a temp file
    that is removed when the context exits.
    """
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        data = read(parsed.canonical)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        try:
            yield Path(tmp.name)
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
    else:
        yield parsed.require_local_path()


def http_url(path: str) -> str:
    """Convert s3:// URIs to a public browser URL."""
    parsed = ParsedURI.parse(path)
    if parsed.scheme == "s3" and parsed.bucket and parsed.key:
        return f"https://{parsed.bucket}.s3.amazonaws.com/{parsed.key}"
    return parsed.canonical if parsed.scheme == "file" else parsed.raw


def is_public_uri(url: str | None) -> bool:
    """Check if a URL is a public HTTP/HTTPS URL."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
