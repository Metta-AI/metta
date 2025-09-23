"""State management for stable release system.

This module handles:
- ReleaseState data model
- State persistence (loading/saving to JSON)
- Git utility functions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from devops.stable.tasks import TaskResult
from metta.common.util.fs import get_repo_root

# ============================================================================
# Helper Functions
# ============================================================================


def get_state_path(version: str) -> Path:
    """Get state file path for a specific release version.

    Args:
        version: Version string

    Returns:
        Path to state file, relative to repo root
    """
    state_dir = get_repo_root() / "devops/stable/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{version}.json"


def get_log_dir(version: str, job_type: str) -> Path:
    """Get log directory for a specific release version and job type.

    Args:
        version: Version string
        job_type: Type of job ("local" or "remote")

    Returns:
        Path to log directory, relative to repo root
    """
    log_dir = get_repo_root() / "devops/stable/logs" / version / job_type
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# ============================================================================
# Data Models
# ============================================================================


class ReleaseState(BaseModel):
    """State of a release qualification run."""

    version: str
    created_at: str
    commit_sha: Optional[str] = None
    results: dict[str, "TaskResult"] = {}
    gates: list[dict] = []
    released: bool = False


# ============================================================================
# State Persistence
# ============================================================================


def load_state(version: str) -> Optional[ReleaseState]:
    """Load release state from JSON file.

    Args:
        version: Version string (with or without 'release_' prefix)

    Returns:
        ReleaseState if found, None otherwise
    """
    path = get_state_path(version)
    if not path.exists():
        return None

    try:
        return ReleaseState.model_validate_json(path.read_text())
    except Exception as e:
        print(f"Failed to load state from {path}: {e}")
        return None


def load_or_create_state(version: str, commit_sha: str) -> ReleaseState:
    """Load existing state or create new one.

    Args:
        version: Version string
        commit_sha: Git commit SHA to use when creating new state

    Returns:
        ReleaseState (either loaded or newly created)
    """
    from datetime import datetime

    state = load_state(version)
    if state:
        return state

    # Create new state
    state = ReleaseState(
        version=version,
        created_at=datetime.utcnow().isoformat(timespec="seconds"),
        commit_sha=commit_sha,
    )
    save_state(state)
    return state


def get_most_recent_state() -> Optional[tuple[str, ReleaseState]]:
    """Get the most recent release state.

    Returns:
        Tuple of (version, state) or None if no state files exist
    """
    state_base = get_repo_root() / "devops/stable/state"
    if not state_base.exists():
        return None

    # Find all .json files
    state_files = []
    for state_file in state_base.glob("*.json"):
        if state_file.is_file():
            # Extract version from filename (remove .json extension)
            version = state_file.stem
            state_files.append((version, state_file))

    if not state_files:
        return None

    # Sort by modification time
    state_files.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    version, _ = state_files[0]

    state = load_state(version)
    if state:
        return (version, state)
    return None


def save_state(state: ReleaseState) -> Path:
    """Save release state to JSON file.

    Note: This is NOT concurrent-safe. Tasks run sequentially, so we don't need
    atomic writes or file locking. If concurrent execution is needed in the future,
    add proper file locking (e.g., via filelock library).
    """
    path = get_state_path(state.version)

    # Simple write - no atomicity needed for sequential execution
    path.write_text(state.model_dump_json(indent=2))

    return path


# Rebuild model after TaskResult is imported to resolve forward references
def _rebuild_models():
    """Rebuild Pydantic models to resolve forward references."""
    from devops.stable.tasks import TaskResult  # noqa: F401

    ReleaseState.model_rebuild()


_rebuild_models()
