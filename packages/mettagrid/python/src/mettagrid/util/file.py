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

from mettagrid.util.uri_resolvers.base import FileParsedScheme, ParsedScheme, S3ParsedScheme
from mettagrid.util.uri_resolvers.schemes import parse_uri


class ParsedURI:
    """Wrapper for backwards compatibility with ParsedScheme."""

    @staticmethod
    def parse(value: str) -> ParsedScheme:
        return parse_uri(value)


logger = logging.getLogger(__name__)


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """Write in-memory bytes/str to *local*, *s3://* destinations."""
    if isinstance(data, str):
        data = data.encode()

    parsed = parse_uri(path, allow_none=False)

    if isinstance(parsed, S3ParsedScheme):
        try:
            boto3.client("s3").put_object(Body=data, Bucket=parsed.bucket, Key=parsed.key, ContentType=content_type)
            logger.debug("Wrote %d B → %s", len(data), http_url(parsed.canonical))
            return
        except NoCredentialsError as e:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise e

    if isinstance(parsed, FileParsedScheme):
        local_path = parsed.local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.debug("Wrote %d B → %s", len(data), local_path)
        return

    raise ValueError(f"Unsupported URI for write_data: {path}")


def exists(path: str) -> bool:
    """Return True if path points to an existing local file or S3 object."""
    parsed = parse_uri(path)

    if isinstance(parsed, S3ParsedScheme):
        try:
            boto3.client("s3").head_object(Bucket=parsed.bucket, Key=parsed.key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "403", "NoSuchKey"}:
                return False
            raise

    if isinstance(parsed, FileParsedScheme):
        return parsed.local_path.exists()

    if parsed.scheme == "mock":
        return True

    return False


def read(path: str) -> bytes:
    """Read bytes from a local path or S3 object."""
    parsed = parse_uri(path)

    if isinstance(parsed, S3ParsedScheme):
        try:
            body = boto3.client("s3").get_object(Bucket=parsed.bucket, Key=parsed.key)["Body"].read()
            logger.debug("Read %d B from %s", len(body), parsed.canonical)
            return body
        except NoCredentialsError:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'", exc_info=True)
            raise

    if isinstance(parsed, FileParsedScheme):
        data = parsed.local_path.read_bytes()
        logger.debug("Read %d B from %s", len(data), parsed.local_path)
        return data

    raise ValueError(f"Unsupported URI for read(): {path}")


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """Upload a file from disk to s3://, or copy locally."""
    parsed = parse_uri(path)

    if isinstance(parsed, S3ParsedScheme):
        boto3.client("s3").upload_file(local_file, parsed.bucket, parsed.key, ExtraArgs={"ContentType": content_type})
        logger.debug("Uploaded %s → %s (size %d B)", local_file, parsed.canonical, os.path.getsize(local_file))
        return

    if isinstance(parsed, FileParsedScheme):
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
    parsed = parse_uri(path)

    if isinstance(parsed, S3ParsedScheme):
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
    elif isinstance(parsed, FileParsedScheme):
        yield parsed.local_path
    else:
        raise ValueError(f"Unsupported URI for local_copy: {path}")


def http_url(path: str) -> str:
    """Convert s3:// URIs to a public browser URL."""
    parsed = parse_uri(path)
    if isinstance(parsed, S3ParsedScheme):
        return f"https://{parsed.bucket}.s3.amazonaws.com/{parsed.key}"
    return parsed.canonical


def _require_local_path(parsed: ParsedScheme) -> Path:
    if isinstance(parsed, FileParsedScheme):
        return parsed.local_path
    raise ValueError(f"URI '{parsed.canonical}' does not refer to a local file path")


def is_public_uri(url: str | None) -> bool:
    """Check if a URL is a public HTTP/HTTPS URL."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
