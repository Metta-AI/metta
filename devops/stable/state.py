"""State management for stable release system.

This module handles:
- ReleaseState data model
- State persistence (loading/saving to JSON)
- Git utility functions
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gitta as git

if TYPE_CHECKING:
    from devops.stable.tasks import TaskResult

# ============================================================================
# Constants
# ============================================================================

STATE_DIR = Path("devops/stable/state")
LOG_DIR_LOCAL = Path("devops/stable/logs/local")
LOG_DIR_REMOTE = Path("devops/stable/logs/remote")

# Create directories
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
LOG_DIR_REMOTE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ReleaseState:
    """State of a release qualification run."""

    version: str
    created_at: str
    commit_sha: Optional[str] = None
    results: dict[str, TaskResult] = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    released: bool = False


# ============================================================================
# Git Utilities
# ============================================================================


def get_commit_sha() -> Optional[str]:
    """Get current git commit SHA."""
    try:
        return git.get_current_commit()
    except git.GitError:
        return None


def get_git_log_since_stable() -> str:
    """Get git log since the last stable release tag."""
    try:
        # Find the latest tag matching v* from HEAD
        last_tag = git.run_git("describe", "--tags", "--abbrev=0", "--match", "v*")

        # Get git log from that tag to HEAD
        return git.run_git("log", f"{last_tag}..HEAD", "--oneline")
    except git.GitError:
        # If no tag found, just return recent commits
        try:
            return git.run_git("log", "--oneline", "-20")
        except git.GitError:
            return "Unable to retrieve git log"


# ============================================================================
# State Persistence
# ============================================================================


def load_state(version: str) -> Optional[ReleaseState]:
    """Load release state from JSON file.

    Args:
        version: Version string (with or without 'release_' prefix)
    """
    from devops.stable.tasks import TaskResult

    # Ensure version has release_ prefix for filename
    if not version.startswith("release_"):
        version = f"release_{version}"

    path = STATE_DIR / f"{version}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())

        # Reconstruct TaskResults
        results = {}
        for name, result_data in data.get("results", {}).items():
            if not isinstance(result_data, dict):
                raise ValueError(f"Invalid result data for {name}")
            # TaskResult doesn't have task_type - remove it if present
            result_data.pop("task_type", None)
            results[name] = TaskResult(**result_data)

        state = ReleaseState(
            version=data["version"],
            created_at=data["created_at"],
            commit_sha=data.get("commit_sha"),
            results=results,
            gates=data.get("gates", []),
            released=data.get("released", False),
        )
        return state
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Failed to load state from {path}: {e}")
        return None


def get_most_recent_state() -> Optional[tuple[str, ReleaseState]]:
    """Get the most recent release state.

    Returns:
        Tuple of (version, state) or None if no state files exist
    """
    state_files = sorted(STATE_DIR.glob("release_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not state_files:
        return None

    # Try to load the most recent state
    most_recent = state_files[0]
    version = most_recent.stem.replace("release_", "")
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
    version = state.version
    # Ensure version has release_ prefix for filename
    if not version.startswith("release_"):
        version = f"release_{version}"

    path = STATE_DIR / f"{version}.json"

    # Convert to dict and serialize
    data = asdict(state)

    # Simple write - no atomicity needed for sequential execution
    path.write_text(json.dumps(data, indent=2, default=str))

    return path
