from __future__ import annotations

import contextlib
import json
import stat
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from metta.common.util.file import local_copy
from mettagrid.policy.policy import PolicySpec


def _resolve_data_path(data_path: Optional[str], extraction_root: Path) -> Optional[str]:
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


def _load_policy_spec(extraction_root: Path) -> PolicySpec:
    """Load a PolicySpec from policy_spec.json in an extracted submission."""
    policy_spec_path = extraction_root / "policy_spec.json"
    if not policy_spec_path.exists():
        raise FileNotFoundError(f"policy_spec.json not found in extracted submission: {extraction_root}")

    with policy_spec_path.open() as f:
        raw_spec = json.load(f)

    spec = PolicySpec.model_validate(raw_spec)
    spec.data_path = _resolve_data_path(spec.data_path, extraction_root)
    return spec


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


@contextmanager
def policy_spec_from_s3_submission(s3_path: str) -> Iterator[PolicySpec]:
    """Download a submission archive from S3 and yield a PolicySpec ready for loading.

    The helper:
    - downloads the archive (supporting s3:// via local_copy)
    - extracts it into a temporary directory
    - reads policy_spec.json and re-roots any relative data_path to the extraction dir
    - prepends the extraction dir to sys.path so custom policy code can be imported
    """
    with tempfile.TemporaryDirectory(prefix="s3_policy_spec_") as tmpdir:
        extraction_root = Path(tmpdir)
        with local_copy(s3_path) as local_archive:
            _extract_submission_archive(local_archive, extraction_root)

        policy_spec = _load_policy_spec(extraction_root)

        sys_path_entry = str(extraction_root.resolve())
        sys.path.insert(0, sys_path_entry)
        try:
            yield policy_spec
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(sys_path_entry)
