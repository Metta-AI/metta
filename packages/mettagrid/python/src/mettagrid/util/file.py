"""
file.py
================
Helpers for reading and writing resources stored on the local filesystem
or in S3 buckets.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

from .uri import ParsedURI

try:  # pragma: no cover - boto3 is optional in some environments
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when S3 is used
    boto3 = None  # type: ignore[assignment]
    ClientError = NoCredentialsError = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
#  Public IO helpers                                                           #
# --------------------------------------------------------------------------- #


def exists(path: str) -> bool:
    """
    Return *True* if *path* points to an existing local file or S3 object.
    """
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        client = _require_s3_client()
        bucket, key = parsed.require_s3()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:  # type: ignore[misc]
            error = getattr(exc, "response", {}).get("Error", {})
            if error.get("Code") in {"404", "403", "NoSuchKey"}:
                return False
            raise
        except NoCredentialsError as exc:  # type: ignore[misc]
            logging.getLogger(__name__).error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise exc

    if parsed.scheme == "file" and parsed.local_path is not None:
        return parsed.local_path.exists()

    if parsed.scheme == "mock":
        # Mock URIs are virtual; treat them as existing.
        return True

    return False


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """Write in-memory bytes/str to local filesystem or S3."""
    logger = logging.getLogger(__name__)

    if isinstance(data, str):
        data = data.encode()

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        client = _require_s3_client()
        bucket, key = parsed.require_s3()
        try:
            client.put_object(Body=data, Bucket=bucket, Key=key, ContentType=content_type)
            logger.info("Wrote %d B → %s", len(data), http_url(parsed.canonical))
            return
        except NoCredentialsError as exc:  # type: ignore[misc]
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise exc

    if parsed.scheme == "file" and parsed.local_path is not None:
        local_path = parsed.local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.info("Wrote %d B → %s", len(data), local_path)
        return

    raise ValueError(f"Unsupported URI for write_data: {path}")


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """Copy a file to local filesystem or upload to S3."""
    logger = logging.getLogger(__name__)

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        client = _require_s3_client()
        bucket, key = parsed.require_s3()
        try:
            extra_args = {"ContentType": content_type} if content_type else None
            if extra_args:
                client.upload_file(local_file, bucket, key, ExtraArgs=extra_args)  # type: ignore[arg-type]
            else:
                client.upload_file(local_file, bucket, key)  # type: ignore[arg-type]
            logger.info(
                "Uploaded %s → %s (size %d B)",
                local_file,
                http_url(parsed.canonical),
                os.path.getsize(local_file),
            )
            return
        except NoCredentialsError as exc:  # type: ignore[misc]
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise exc

    if parsed.scheme == "file" and parsed.local_path is not None:
        dst = parsed.local_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dst)
        logger.info("Copied %s → %s (size %d B)", local_file, dst, Path(local_file).stat().st_size)
        return

    raise ValueError(f"Unsupported URI for write_file: {path}")


def read(path: str) -> bytes:
    """Read bytes from a local path or S3 object."""
    logger = logging.getLogger(__name__)

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        client = _require_s3_client()
        bucket, key = parsed.require_s3()
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            logger.info("Read %d B from %s", len(body), parsed.canonical)
            return body
        except NoCredentialsError as exc:  # type: ignore[misc]
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise exc

    if parsed.scheme == "file" and parsed.local_path is not None:
        data = parsed.local_path.read_bytes()
        logger.info("Read %d B from %s", len(data), parsed.local_path)
        return data

    raise ValueError(f"Unsupported URI for read(): {path}")


@contextmanager
def local_copy(path: str) -> Iterator[Path]:
    """
    Yield a local *Path* for *path* (supports local paths and S3 objects).

    Local paths are yielded as-is. S3 objects are streamed into a temporary
    file that is cleaned up when the context exits.
    """
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "s3":
        with _temporary_path() as tmp_path:
            _download_s3_to_path(parsed, tmp_path)
            yield tmp_path
        return

    yield parsed.require_local_path()


def http_url(path: str) -> str:
    """Return a public HTTP URL when available."""
    parsed = ParsedURI.parse(path)
    if parsed.scheme == "s3" and parsed.bucket and parsed.key:
        return f"https://{parsed.bucket}.s3.amazonaws.com/{parsed.key}"
    return parsed.canonical if parsed.scheme == "file" else parsed.raw


def is_public_uri(url: str | None) -> bool:
    """Check if a URL is a public HTTP/HTTPS URL."""
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _require_s3_client() -> Any:
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 operations but is not installed")
    return boto3.client("s3")


@contextlib.contextmanager
def _temporary_path() -> Iterator[Path]:
    fd, name = tempfile.mkstemp(prefix="metta_s3_")
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def _download_s3_to_path(parsed: ParsedURI, destination: Path) -> None:
    client = _require_s3_client()
    bucket, key = parsed.require_s3()
    try:
        client.download_file(bucket, key, str(destination))
    except NoCredentialsError as exc:  # type: ignore[misc]
        logging.getLogger(__name__).error("AWS credentials not found; run 'aws sso login --profile softmax'")
        raise exc
