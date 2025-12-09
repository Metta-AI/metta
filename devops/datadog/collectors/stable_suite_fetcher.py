"""Data fetching for stable_suite JobState database.

This module handles reading job state from stable_suite's SQLite database.
Reads from local filesystem: devops/stable/state/{version}/jobs.sqlite
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from sqlmodel import Session, create_engine, select

from metta.common.util.fs import get_repo_root
from metta.jobs.job_state import JobState, JobStatus

logger = logging.getLogger(__name__)

STABLE_STATE_BASE = Path("devops/stable/state")


def find_latest_version() -> Optional[str]:
    """Find the most recent stable_suite version.

    Returns:
        Version string (e.g., "v0.1.0"), or None if no versions found
    """
    state_dir = get_repo_root() / STABLE_STATE_BASE
    if not state_dir.exists():
        logger.warning("Stable state directory does not exist: %s", state_dir)
        return None

    version_dirs = [d for d in state_dir.iterdir() if d.is_dir()]
    if not version_dirs:
        logger.warning("No version directories found in %s", state_dir)
        return None

    # Sort by modification time, get most recent
    latest = max(version_dirs, key=lambda d: d.stat().st_mtime)
    version = latest.name
    logger.info("Found latest version: %s", version)
    return version


def get_latest_jobs_db_path() -> Optional[str]:
    """Get path to the latest version's jobs.sqlite database.

    Returns:
        Path to jobs.sqlite file, or None if not found
    """
    version = find_latest_version()
    if version is None:
        return None

    return get_jobs_db_path_for_version(version)


def get_jobs_db_path_for_version(version: str) -> Optional[str]:
    """Get path to jobs.sqlite for a specific version.

    Args:
        version: Version string (e.g., "v0.1.0")

    Returns:
        Path to jobs.sqlite file, or None if not found
    """
    state_dir = get_repo_root() / STABLE_STATE_BASE
    db_path = state_dir / version / "jobs.sqlite"

    if not db_path.exists():
        logger.warning("jobs.sqlite not found at %s", db_path)
        return None

    return str(db_path)


def read_job_states(db_path: str, since_days: int = 1) -> List[JobState]:
    """Load JobState entries from database.

    Args:
        db_path: Path to jobs.sqlite database
        since_days: Only return jobs completed within last N days (default: 1)

    Returns:
        List of JobState objects

    Raises:
        FileNotFoundError: If database file doesn't exist
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    cutoff_iso = cutoff.isoformat()

    with Session(engine) as session:
        # Query for completed jobs within time window
        statement = select(JobState).where(
            JobState.status == JobStatus.COMPLETED,
            JobState.completed_at >= cutoff_iso,
        )
        results = session.exec(statement).all()
        jobs = list(results)
        logger.info("Found %d completed jobs since %s", len(jobs), cutoff_iso)
        return jobs


def get_latest_job_states(since_days: int = 1) -> List[JobState]:
    """Get job states from the latest version's database.

    Args:
        since_days: Only return jobs completed within last N days (default: 1)

    Returns:
        List of JobState objects, or empty list if database not found
    """
    db_path = get_latest_jobs_db_path()
    if db_path is None:
        logger.warning("No jobs.sqlite database found")
        return []

    try:
        return read_job_states(db_path, since_days=since_days)
    except Exception as exc:
        logger.error("Failed to read job states from %s: %s", db_path, exc, exc_info=True)
        return []
