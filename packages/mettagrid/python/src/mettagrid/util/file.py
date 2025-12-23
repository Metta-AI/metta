from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from mettagrid.util.uri_resolvers.schemes import parse_uri

logger = logging.getLogger(__name__)


def copy_data(src: str, dest: str, content_type: str = "application/octet-stream") -> None:
    data = read(src)
    write_data(dest, data, content_type=content_type)


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """Write in-memory bytes/str to *local*, *s3://* destinations."""
    if isinstance(data, str):
        data = data.encode()

    parsed = parse_uri(path, allow_none=False)

    if parsed.scheme == "s3":
        try:
            boto3.client("s3").put_object(Body=data, Bucket=parsed.bucket, Key=parsed.key, ContentType=content_type)
            logger.debug("Wrote %d B → %s", len(data), http_url(parsed.canonical))
            return
        except NoCredentialsError as e:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise e

    if parsed.scheme == "file":
        local_path = parsed.local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.debug("Wrote %d B → %s", len(data), local_path)
        return

    raise ValueError(f"Unsupported URI for write_data: {path}")


def exists(path: str) -> bool:
    """Return True if path points to an existing local file or S3 object."""
    parsed = parse_uri(path, allow_none=True)
    if parsed is None:
        return False

    if parsed.scheme == "s3":
        try:
            boto3.client("s3").head_object(Bucket=parsed.bucket, Key=parsed.key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "403", "NoSuchKey"}:
                return False
            raise

    if parsed.scheme == "file":
        return parsed.local_path.exists()

    if parsed.scheme == "mock":
        return True

    return False


def read(path: str) -> bytes:
    """Read bytes from a local path or S3 object."""
    parsed = parse_uri(path, allow_none=False)

    if parsed.scheme == "s3":
        try:
            body = boto3.client("s3").get_object(Bucket=parsed.bucket, Key=parsed.key)["Body"].read()
            logger.debug("Read %d B from %s", len(body), parsed.canonical)
            return body
        except NoCredentialsError:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise

    if parsed.scheme == "file":
        data = parsed.local_path.read_bytes()
        logger.debug("Read %d B from %s", len(data), parsed.local_path)
        return data

    raise ValueError(f"Unsupported URI for read(): {path}")


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """Upload a file from disk to s3://, or copy locally."""
    parsed = parse_uri(path, allow_none=False)

    if parsed.scheme == "s3":
        boto3.client("s3").upload_file(local_file, parsed.bucket, parsed.key, ExtraArgs={"ContentType": content_type})
        logger.debug("Uploaded %s → %s (size %d B)", local_file, parsed.canonical, os.path.getsize(local_file))
        return

    if parsed.scheme == "file":
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
    parsed = parse_uri(path, allow_none=False)

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
    elif parsed.scheme == "file":
        yield parsed.local_path
    else:
        raise ValueError(f"Unsupported URI for local_copy: {path}")


def http_url(path: str) -> str:
    """Convert s3:// URIs to a public browser URL."""
    parsed = parse_uri(path, allow_none=False)
    if parsed.scheme == "s3":
        return f"https://{parsed.bucket}.s3.amazonaws.com/{parsed.key}"
    return parsed.canonical


def is_public_uri(url: str | None) -> bool:
    """Check if a URL is a public HTTP/HTTPS URL."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
