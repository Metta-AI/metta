"""Utilities for preparing PolicySpec from remote sources."""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from typing import Optional

from mettagrid.policy.policy import PolicySpec
from mettagrid.util.file import local_copy

DEFAULT_POLICY_CACHE_DIR = Path("/tmp/mettagrid-policy-cache")

_registered_cleanup_dirs: set[Path] = set()


def _validate_archive_member(entry: zipfile.ZipInfo, destination_root: Path) -> None:
    """Ensure a zip entry extracts within destination_root without using symlinks."""
    member_path = Path(entry.filename)

    if member_path.is_absolute():
        raise ValueError(f"Submission archive contains absolute path: {entry.filename}")
    if ".." in member_path.parts:
        raise ValueError(f"Submission archive contains path traversal: {entry.filename}")
    if stat.S_ISLNK(entry.external_attr >> 16):
        raise ValueError(f"Submission archive contains symlink entry: {entry.filename}")

    target_path = (destination_root / member_path).resolve()
    if destination_root != target_path and destination_root not in target_path.parents:
        raise ValueError(f"Submission archive entry escapes extraction directory: {entry.filename}")


def _extract_submission_archive(archive_path: Path, destination: Path) -> None:
    """Extract a submission archive into destination."""
    destination_root = destination.resolve()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for entry in archive.infolist():
                _validate_archive_member(entry, destination_root)
            archive.extractall(destination_root)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid submission archive: {archive_path}") from exc


def _resolve_spec_data_path(data_path: Optional[str], extraction_root: Path) -> Optional[str]:
    """Resolve a policy data path inside the extracted submission."""
    if data_path is None:
        return None

    candidate = Path(data_path)
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Policy data path does not exist: {candidate}")

    resolved = extraction_root / candidate
    if resolved.exists():
        return str(resolved)

    raise FileNotFoundError(f"Policy data path '{data_path}' not found in submission directory {extraction_root}")


def load_policy_spec_from_local_dir(
    extraction_root: Path,
    *,
    device: str | None = None,
) -> PolicySpec:
    """Load a PolicySpec from policy_spec.json in an extracted submission."""
    policy_spec_path = extraction_root / "policy_spec.json"
    if not policy_spec_path.exists():
        raise FileNotFoundError(f"policy_spec.json not found in extracted submission: {extraction_root}")

    with policy_spec_path.open() as f:
        raw_spec = json.load(f)

    spec = PolicySpec.model_validate(raw_spec)
    spec.data_path = _resolve_spec_data_path(spec.data_path, extraction_root)
    if device is not None and "device" in spec.init_kwargs:
        spec.init_kwargs["device"] = device
    sys_path_entry = str(extraction_root.resolve())
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)
    return spec


def _cleanup_cache_dir(cache_dir: Path) -> None:
    """Atexit handler to clean up a cache directory."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def load_policy_spec_from_s3(
    s3_path: str,
    cache_dir: Optional[Path] = None,
    remove_downloaded_copy_on_exit: bool = False,
    *,
    device: str | None = None,
) -> PolicySpec:
    """Download a submission archive from S3 and return a PolicySpec ready for loading.

    Downloads the archive to a deterministic cache location based on the URI hash,
    allowing reuse across calls with the same URI.

    Args:
        s3_path: S3 path to the submission archive (e.g., s3://bucket/path/submission.zip)
        cache_dir: Base directory for caching. Defaults to /tmp/mettagrid-policy-cache
        cleanup_on_exit: If True, register an atexit handler to clean up the cache directory
        device: Override the device in the loaded spec (e.g., "cpu" or "cuda:0")

    Returns:
        PolicySpec with paths resolved to the local extraction directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_POLICY_CACHE_DIR

    extraction_root = cache_dir / hashlib.sha256(s3_path.encode()).hexdigest()[:16]
    marker_file = extraction_root / ".extraction_complete"

    if not marker_file.exists():
        extraction_root.mkdir(parents=True, exist_ok=True)

        with local_copy(s3_path) as local_archive:
            _extract_submission_archive(local_archive, extraction_root)

        marker_file.touch()

    policy_spec = load_policy_spec_from_local_dir(extraction_root, device=device)

    if remove_downloaded_copy_on_exit and extraction_root not in _registered_cleanup_dirs:
        _registered_cleanup_dirs.add(extraction_root)
        atexit.register(_cleanup_cache_dir, extraction_root)

    return policy_spec
