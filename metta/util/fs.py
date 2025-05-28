import os
from pathlib import Path


def cd_repo_root():
    """
    Ensure we're running in the repository root.

    Raises:
        SystemExit: If repository root cannot be found
    """
    current = Path.cwd().resolve()
    search_paths = [current] + list(current.parents)

    for parent in search_paths:
        if (parent / ".git").exists():
            os.chdir(parent)
            return

    # If we get here, no .git directory was found
    raise SystemExit("Repository root not found - no .git directory in current path or parent directories")
