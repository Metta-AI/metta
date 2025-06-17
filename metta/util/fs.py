import os
from pathlib import Path
from typing import Optional


def get_repo_root(markers: Optional[list[str]] = None) -> Path:
    """
    Get the repository root directory by searching for common root markers.

    Args:
        markers: List of files/directories that indicate repo root.
                 Defaults to ['.git', 'pyproject.toml', 'setup.py', 'configs']

    Returns:
        Path: The repository root directory

    Raises:
        ValueError: If repo root cannot be determined
    """
    if markers is None:
        markers = [".git", "pyproject.toml", "setup.py", "configs"]

    current = Path.cwd().resolve()
    search_paths = [current] + list(current.parents)

    for parent in search_paths:
        if any((parent / marker).exists() for marker in markers):
            return parent

    raise ValueError(
        f"Could not determine repository root. Searched for markers {markers} in {current} and parent directories."
    )


def cd_repo_root():
    """
    Ensure we're running in the repository root.

    Raises:
        SystemExit: If repository root cannot be found
    """
    try:
        repo_root = get_repo_root()
        os.chdir(repo_root)
    except ValueError as e:
        raise SystemExit(str(e)) from e
