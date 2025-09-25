"""File operations for mettagrid - local files only.

This module provides basic file operations for mettagrid without external dependencies.
For advanced S3 and remote file operations, use metta.utils.file instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read(path: str) -> bytes:
    """Read bytes from a local file path.

    Note: This only supports local files. For S3 support, use metta.utils.file.
    """
    if path.startswith("s3://"):
        raise NotImplementedError("S3 support requires metta.utils.file. This is a simple local-only implementation.")

    file_path = Path(path).expanduser().resolve()
    data = file_path.read_bytes()
    logger.info("Read %d B from %s", len(data), file_path)
    return data


def write_data(path: str, data: str | bytes, *, content_type: str = "application/octet-stream") -> None:
    """Write data to a local file path.

    Note: This only supports local files. For S3 support, use metta.utils.file.
    """
    if path.startswith("s3://"):
        raise NotImplementedError("S3 support requires metta.utils.file. This is a simple local-only implementation.")

    if isinstance(data, str):
        data = data.encode()

    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data)
    logger.info("Wrote %d B to %s", len(data), file_path)
