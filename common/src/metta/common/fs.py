import os
import subprocess
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Returns:
        Path to the repository root

    Raises:
        SystemExit: If repository root cannot be found
    """
    current = Path.cwd().resolve()
    search_paths = [current] + list(current.parents)

    for parent in search_paths:
        if (parent / ".git").exists():
            return parent

    # If we get here, no .git directory was found
    raise SystemExit("Repository root not found - no .git directory in current path or parent directories")


def cd_repo_root():
    """
    Ensure we're running in the repository root.

    Raises:
        SystemExit: If repository root cannot be found
    """
    repo_root = get_repo_root()
    os.chdir(repo_root)


def tree(path: Optional[Path] = None, max_depth: Optional[int] = None) -> str:
    """
    Collect the output of the tree command.

    Args:
        path: Directory path to tree (defaults to current directory)
        max_depth: Maximum depth to traverse

    Returns:
        Tree command output as string

    Raises:
        FileNotFoundError: If tree command is not available
        subprocess.CalledProcessError: If tree command fails
    """
    cmd = ["tree"]

    # Add path if specified
    if path:
        cmd.append(str(path))

    # Add depth limit
    if max_depth is not None:
        cmd.extend(["-L", str(max_depth)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "tree command not found. Install it with: brew install tree (macOS) or apt install tree (Ubuntu/Debian)"
        ) from e
