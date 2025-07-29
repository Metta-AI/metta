"""Git filter-repo utilities for extracting repository subsets."""

import subprocess
import tempfile
from pathlib import Path

from .git import GitError, add_remote, get_commit_count, get_file_list, run_git, run_git_in_dir


def filter_repo(source_path: Path, paths: list[str]) -> Path:
    """Filter repository to only include specified paths.

    Args:
        source_path: Path to source repository
        paths: List of paths to keep (e.g., ["mettagrid/", "mettascope/"])

    Returns:
        Path to the filtered repository
    """
    if not (source_path / ".git").exists():
        raise ValueError(f"Not a git repository: {source_path}")

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

    # Check if git-filter-repo is available
    try:
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "git-filter-repo not found. Install with:\n"
            "curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo\n"
            "chmod +x git-filter-repo && sudo mv git-filter-repo /usr/local/bin/"
        ) from e

    # Filter repository
    filter_cmd = ["git", "filter-repo", "--force"]
    for path in paths:
        filter_cmd.extend(["--path", path])

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
