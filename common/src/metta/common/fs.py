import os
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


def _python_tree(path: Path, max_depth: Optional[int] = None, _current_depth: int = 0, _prefix: str = "") -> str:
    """
    Python-based tree implementation as fallback.

    Args:
        path: Directory path to tree
        max_depth: Maximum depth to traverse
        _current_depth: Current recursion depth (internal use)
        _prefix: Current line prefix (internal use)

    Returns:
        Tree-like string representation
    """
    if not path.exists():
        return f"{path} [does not exist]\n"

    if not path.is_dir():
        return f"{path} [not a directory]\n"

    result = ""
    if _current_depth == 0:
        result += f"{path}\n"

    # Stop if we've reached max depth
    if max_depth is not None and _current_depth >= max_depth:
        return result

    try:
        # Get all items in directory, sorted
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

        for i, item in enumerate(items):
            is_last = i == len(items) - 1

            # Create the tree symbols
            if is_last:
                current_prefix = _prefix + "└── "
                next_prefix = _prefix + "    "
            else:
                current_prefix = _prefix + "├── "
                next_prefix = _prefix + "│   "

            result += current_prefix + item.name

            if item.is_dir():
                result += "/\n"
                # Recurse into subdirectory
                if max_depth is None or _current_depth + 1 < max_depth:
                    result += _python_tree(item, max_depth, _current_depth + 1, next_prefix)
            else:
                result += "\n"

    except PermissionError:
        result += f"{_prefix}[Permission Denied]\n"
    except Exception as e:
        result += f"{_prefix}[Error: {e}]\n"

    return result


def tree(path: Optional[Path] = None, max_depth: Optional[int] = None) -> str:
    """
    Generate a tree-like directory listing using pure Python.

    Args:
        path: Directory path to tree (defaults to current directory)
        max_depth: Maximum depth to traverse

    Returns:
        Tree-like string representation of directory structure
    """
    # Use current directory if no path specified
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    return _python_tree(path, max_depth)
