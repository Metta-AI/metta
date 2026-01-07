"""Utilities for preparing PolicySpec from remote sources."""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import logging
import os
import secrets
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec
from mettagrid.util.file import read as s3_read

logger = logging.getLogger(__name__)

DEFAULT_POLICY_CACHE_DIR = Path("/tmp/mettagrid-policy-cache")

_registered_cleanup_dirs: set[Path] = set()
_registered_cleanup_files: set[Path] = set()


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def _setup_marker_paths(extraction_root: Path, setup_script: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(setup_script.encode()).hexdigest()[:16]
    return (extraction_root / f".setup-{digest}.lock", extraction_root / f".setup-{digest}.done")


def _ensure_setup_script_ran(setup_script: str, extraction_root: Path) -> None:
    lock_path, done_path = _setup_marker_paths(extraction_root, setup_script)

    with _exclusive_file_lock(lock_path):
        if done_path.exists():
            return

        _run_setup_script(extraction_root / setup_script, extraction_root)
        done_path.touch()


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

    candidate = Path(data_path).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Policy data path does not exist: {candidate}")

    resolved = extraction_root / candidate
    if resolved.exists():
        return str(resolved.resolve())

    raise FileNotFoundError(f"Policy data path '{data_path}' not found in submission directory {extraction_root}")


def _find_package_source_root(extraction_root: Path, class_path: str) -> Path | None:
    """Find the source root by locating the top-level package directory.

    Given a class_path like 'mypackage.submodule.MyClass', finds a directory named
    'mypackage' that contains Python code, and returns its parent (the source root).

    Note: This modifies sys.path but does not invalidate sys.modules. If the same
    module was previously imported from a different location (e.g., installed package),
    Python will use the cached import. This is acceptable for remote evaluation where
    each task runs in a fresh process, but may cause issues in long-running processes
    that load multiple submissions with the same class_path.
    """
    top_package = class_path.split(".")[0]

    # Find any __init__.py inside a directory named after the top package
    # e.g., for "cogames.policy.module", find "**/cogames/**/__init__.py"
    for init_file in extraction_root.rglob("__init__.py"):
        if "__pycache__" in str(init_file):
            continue
        # Check if any ancestor directory is named after the top package
        for parent in init_file.parents:
            if parent.name == top_package and parent != extraction_root:
                # Found it - source root is the parent of the package directory
                return parent.parent

    return None


def _run_setup_script(setup_script_path: Path, extraction_root: Path) -> None:
    """Run a setup script from the submission archive.

    The script is executed with the extraction root as the working directory.
    """
    if not setup_script_path.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_script_path}")

    if not setup_script_path.suffix == ".py":
        raise ValueError(f"Setup script must be a .py file: {setup_script_path}")

    logger.info("Running setup script: %s", setup_script_path)

    result = subprocess.run(
        [sys.executable, str(setup_script_path)],
        cwd=extraction_root,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Setup script failed with exit code {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    logger.info("Setup script completed successfully")


_executed_setup_scripts: set[Path] = set()


def _cleanup_cache_dir(cache_dir: Path) -> None:
    """Atexit handler to clean up a cache directory."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _schedule_cleanup_cache_file(path: Path) -> None:
    if path not in _registered_cleanup_files:
        _registered_cleanup_files.add(path)
        atexit.register(_cleanup_cache_file, path)


def _cleanup_cache_file(path: Path) -> None:
    """atexit handler to clean up a single file."""
    if path.exists():
        os.remove(path)


def download_policy_spec_from_s3_as_zip(
    s3_path: str,
    cache_dir: Optional[Path] = None,
    remove_downloaded_copy_on_exit: bool = False,
) -> Path:
    """Download a submission archive from S3 without extracting.

    Args:
        s3_path: S3 path to a submission archive (.zip)
        cache_dir: Base directory for caching. Defaults to /tmp/mettagrid-policy-cache
        remove_downloaded_copy_on_exit: If True, register an atexit handler to clean up the cache entry

    Returns:
        Local path to the downloaded zip file.
    """
    cache_dir = cache_dir or DEFAULT_POLICY_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = s3_path.rstrip("/")
    if not normalized_path.endswith(".zip"):
        raise ValueError("Expected a .zip submission archive.")
    digest = hashlib.sha256(normalized_path.encode()).hexdigest()
    tmp_local_path = cache_dir / f"tmp-{digest}-{secrets.token_hex(8)}.zip"
    local_path = cache_dir / f"{digest}.zip"

    if local_path.exists():
        return local_path

    _schedule_cleanup_cache_file(tmp_local_path)
    if remove_downloaded_copy_on_exit:
        _schedule_cleanup_cache_file(local_path)

    # download at a temporary path and use atomic rename so we don't see partial results
    with open(tmp_local_path, mode="wb") as f:
        f.write(s3_read(normalized_path))
    os.rename(tmp_local_path, local_path)

    return local_path


def load_policy_spec_from_path(
    local_path: Path,
    *,
    device: str | None = None,
    remove_downloaded_copy_on_exit: bool = False,
    force_dest: Optional[Path] = None,
) -> PolicySpec:
    if local_path.is_dir():
        extraction_root = local_path
    else:
        extraction_root = force_dest or (
            DEFAULT_POLICY_CACHE_DIR / hashlib.sha256(local_path.as_uri().encode()).hexdigest()
        ).with_suffix(".d")
        extraction_root.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(extraction_root / ".extraction.lock"):
            if not (extraction_root / ".extraction_complete").exists():
                _extract_submission_archive(local_path, extraction_root)
                (extraction_root / ".extraction_complete").touch()

                if remove_downloaded_copy_on_exit and extraction_root not in _registered_cleanup_dirs:
                    _registered_cleanup_dirs.add(extraction_root)
                    atexit.register(_cleanup_cache_dir, extraction_root)

    policy_spec_path = extraction_root / POLICY_SPEC_FILENAME
    if not policy_spec_path.exists():
        raise FileNotFoundError(f"{POLICY_SPEC_FILENAME} not found in extracted submission: {extraction_root}")

    submission_spec = SubmissionPolicySpec.model_validate_json(policy_spec_path.read_text())

    if submission_spec.setup_script and extraction_root not in _executed_setup_scripts:
        _ensure_setup_script_ran(submission_spec.setup_script, extraction_root)
        _executed_setup_scripts.add(extraction_root)

    spec = PolicySpec(
        class_path=submission_spec.class_path,
        data_path=submission_spec.data_path,
        init_kwargs=submission_spec.init_kwargs,
    )
    spec.data_path = _resolve_spec_data_path(spec.data_path, extraction_root)
    if device is not None and "device" in spec.init_kwargs:
        spec.init_kwargs["device"] = device

    module_root = _find_package_source_root(extraction_root, spec.class_path)
    if module_root and module_root != extraction_root:
        sys_path_entry = str(module_root.resolve())
        if sys_path_entry not in sys.path:
            sys.path.insert(0, sys_path_entry)

    sys_path_entry = str(extraction_root.resolve())
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)

    return spec
