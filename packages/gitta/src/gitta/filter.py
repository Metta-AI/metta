"""Git filter-repo functionality."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .core import GitError, run_git
from .git import get_commit_count, get_file_list


def filter_repo(source_path: Path, paths: List[str], make_root: Optional[str] = None) -> Path:
    """Filter repository to only include specified paths.

    Args:
        source_path: Path to source repository
        paths: List of paths to keep (e.g., ["packages/mettagrid/", "mettascope/"])
        make_root: If provided, the matched path will be moved to repo root (e.g., "packages/mettagrid/")

    Returns:
        Path to the filtered repository
    """

    if not (source_path / ".git").exists():
        raise ValueError(f"Not a git repository: {source_path}")

    # Check if git-filter-repo is available before doing any cloning work
    try:
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Use the same installation method as metta CLI tool
        raise RuntimeError(
            "\ngit-filter-repo not found. Please install it:\n\n"
            "  metta install filter-repo\n\n"
            "Or install manually:\n"
            "  curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo\n"
            "  chmod +x git-filter-repo\n"
            "  sudo mv git-filter-repo /usr/local/bin/\n"
        ) from e

    # Create temporary directory
    target_dir = Path(tempfile.mkdtemp(prefix="filtered-repo-"))
    filtered_path = target_dir / "filtered"

    print("Cloning for filtering...")

    # Clone locally
    source_url = f"file://{source_path.absolute()}"
    try:
        run_git("clone", "--no-local", source_url, str(filtered_path))
    except GitError as e:
        raise RuntimeError(f"Failed to clone: {e}") from e

    # Filter repository
    filter_cmd = ["git", "filter-repo", "--force"]
    for path in paths:
        filter_cmd.extend(["--path", path])

    if make_root:
        filter_cmd.extend(["--path-rename", f"{make_root}:"])
        print(f"Filtering to: {', '.join(paths)}, moving {make_root} to root")
    else:
        print(f"Filtering to: {', '.join(paths)}")

    result = subprocess.run(filter_cmd, cwd=filtered_path, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git-filter-repo failed: {result.stderr.strip()}")

    # Verify result
    files = get_file_list(filtered_path)

    if not files:
        raise RuntimeError("Filtered repository is empty!")

    commit_count = get_commit_count(filtered_path)
    print(f"✅ Filtered: {len(files)} files, {commit_count} commits")

    return filtered_path
