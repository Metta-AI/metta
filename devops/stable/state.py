"""State management for stable release system.

This module handles:
- ReleaseState data model
- State persistence (loading/saving to JSON)
- Git utility functions
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from metta.common.util.fs import get_repo_root

# ============================================================================
# Helper Functions
# ============================================================================


def get_state_path(version: str) -> Path:
    state_dir = get_repo_root() / "devops/stable/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{version}.json"


# ============================================================================
# State Persistence
# ============================================================================


class Gate(BaseModel):
    """Tracks completion of pipeline-level gates (prepare_tag, bug_check, etc.)."""

    step: str
    passed: bool
    timestamp: str


class ReleaseState(BaseModel):
    version: str
    created_at: str
    commit_sha: Optional[str] = None
    gates: list[Gate] = Field(default_factory=list)
    released: bool = False


def load_state(version: str) -> Optional[ReleaseState]:
    path = get_state_path(version)
    if not path.exists():
        return None

    try:
        return ReleaseState.model_validate_json(path.read_text())
    except Exception as e:
        print(f"Failed to load state from {path}: {e}")
        return None


def load_or_create_state(version: str, commit_sha: str) -> ReleaseState:
    state = load_state(version)
    if state:
        return state

    # Create new state
    state = ReleaseState(
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit_sha=commit_sha,
    )
    save_state(state)
    return state


def get_most_recent_state() -> Optional[tuple[str, ReleaseState]]:
    """
    Only considers state files with valid timestamp-based version format vYYYY.MM.DD-HHMMSS.json
    Also accepts legacy 4-digit format vYYYY.MM.DD-HHMM.json for backward compatibility.
    """
    state_base = get_repo_root() / "devops/stable/state"
    if not state_base.exists():
        return None

    # Pattern for valid timestamp-based version format: vYYYY.MM.DD-HHMMSS or vYYYY.MM.DD-HHMM
    version_pattern = re.compile(r"^v\d{4}\.\d{2}\.\d{2}-\d{4,6}$")

    # Find all .json files with valid version format
    state_files = []
    for state_file in state_base.glob("*.json"):
        if state_file.is_file():
            # Extract version from filename (remove .json extension)
            version = state_file.stem
            # Only include files with valid timestamp-based version format
            if version_pattern.match(version):
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
    path = get_state_path(state.version)

    # Simple write - no atomicity needed for sequential execution
    path.write_text(state.model_dump_json(indent=2))

    return path
