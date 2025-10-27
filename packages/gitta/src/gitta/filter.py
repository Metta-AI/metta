"""Git repository filtering using built-in git commands."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .core import GitError, run_git
from .git import get_commit_count, get_file_list


def filter_repo(source_path: Path, paths: list[str], make_root: str | None = None) -> Path:
    """Filter repository to only include specified paths using git subtree split.

    This uses git's built-in subtree split command to extract a subdirectory
    with its full commit history, making it the root of a new repository.

    Args:
        source_path: Path to source repository
        paths: List of paths to keep (currently only single path supported)
        make_root: The path to extract and make the repository root

    Returns:
        Path to the filtered repository

    Raises:
        ValueError: If source is not a git repository or paths configuration is invalid
        RuntimeError: If git operations fail
    """
    if not (source_path / ".git").exists():
        raise ValueError(f"Not a git repository: {source_path}")

    if not make_root:
        raise ValueError(
            "filter_repo requires make_root parameter when using git subtree split. "
            "Specify which directory to extract as the root."
        )

    if len(paths) > 1:
        raise ValueError(
            "filter_repo currently only supports a single path when using git subtree split. "
            f"Got {len(paths)} paths: {paths}"
        )

    # The directory to extract (without trailing slash)
    prefix = make_root.rstrip("/")

    # Ensure the path exists in the repository

    full_path = source_path / prefix
    if not full_path.exists() or not full_path.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {prefix}")

    print(f"Extracting {prefix} using git subtree split...")

    # Create a temporary branch name
    branch_name = f"subtree-split-{int(time.time())}"

    try:
        # Use git subtree split to create a branch with just this subdirectory's history
        # This preserves all commit history for files in this directory
        run_git(
            "-C",
            str(source_path),
            "subtree",
            "split",
            "--prefix",
            prefix,
            "-b",
            branch_name,
        )
    except GitError as e:
        raise RuntimeError(f"Failed to split subtree: {e}") from e

    # Create temporary directory for the filtered repository
    target_dir = Path(tempfile.mkdtemp(prefix="filtered-repo-"))
    filtered_path = target_dir / "filtered"

    try:
        # Clone the split branch to our target location
        # This gives us a clean repository with the subdirectory as root
        # Use -C to run from a stable directory (not the test's temp dir)
        run_git(
            "-C",
            str(target_dir),
            "clone",
            "--branch",
            branch_name,
            "--single-branch",
            str(source_path),
            "filtered",
        )
    except GitError as e:
        raise RuntimeError(f"Failed to clone filtered branch: {e}") from e
    finally:
        # Clean up the temporary branch
        try:
            run_git("-C", str(source_path), "branch", "-D", branch_name)
        except GitError:
            # If cleanup fails, that's not critical - continue
            pass

    # Verify result
    files = get_file_list(filtered_path)

    if not files:
        raise RuntimeError("Filtered repository is empty!")

    commit_count = get_commit_count(filtered_path)
    print(f"✅ Filtered: {len(files)} files, {commit_count} commits")

    return filtered_path
