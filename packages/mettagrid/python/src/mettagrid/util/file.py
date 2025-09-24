"""
file.py
================
Read and write files to local filesystem.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from .uri import ParsedURI

# --------------------------------------------------------------------------- #
#  Public IO helpers                                                           #
# --------------------------------------------------------------------------- #


def exists(path: str) -> bool:
    """
    Return *True* if *path* points to an existing local file.
    """
    parsed = ParsedURI.parse(path)

    if parsed.scheme == "file" and parsed.local_path is not None:
        return parsed.local_path.exists()

    if parsed.scheme == "mock":
        # Mock URIs are virtual; treat them as existing.
        return True

    return False


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """Write in-memory bytes/str to local filesystem."""
    logger = logging.getLogger(__name__)

    if isinstance(data, str):
        data = data.encode()

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "file" and parsed.local_path is not None:
        local_path = parsed.local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.info("Wrote %d B → %s", len(data), local_path)
        return

    raise ValueError(f"Unsupported URI for write_data: {path}")


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """Copy a file to local filesystem."""
    logger = logging.getLogger(__name__)

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "file" and parsed.local_path is not None:
        dst = parsed.local_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dst)
        logger.info("Copied %s → %s (size %d B)", local_file, dst, Path(local_file).stat().st_size)
        return

    raise ValueError(f"Unsupported URI for write_file: {path}")


def read(path: str) -> bytes:
    """Read bytes from a local path."""
    logger = logging.getLogger(__name__)

    parsed = ParsedURI.parse(path)

    if parsed.scheme == "file" and parsed.local_path is not None:
        data = parsed.local_path.read_bytes()
        logger.info("Read %d B from %s", len(data), parsed.local_path)
        return data

    raise ValueError(f"Unsupported URI for read(): {path}")


@contextmanager
def local_copy(path: str):
    """
    Yield a local *Path* for *path* (supports local paths).

    Local paths are yielded as-is.

    Usage:
        with local_copy(uri) as p:
            do_something_with(Path(p))
    """
    parsed = ParsedURI.parse(path)
    yield parsed.require_local_path()


def http_url(path: str) -> str:
    """Return the path as is for local files."""
    parsed = ParsedURI.parse(path)
    return parsed.canonical if parsed.scheme == "file" else parsed.raw


def is_public_uri(url: str | None) -> bool:
    """
    Check if a URL is a public HTTP/HTTPS URL.
    """
    if not url:
        return False
    # Simple check for http/https URLs
    return url.startswith("http://") or url.startswith("https://")
