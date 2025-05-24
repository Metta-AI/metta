"""Filesystem utilities for the Metta project."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from metta.util.colorama import green, red, yellow


def find_repo_root(
    start_path: Optional[Union[str, Path]] = None, git_marker: bool = True, custom_markers: Optional[List[tuple]] = None
) -> Optional[Path]:
    """
    Find the repository root by looking for .git directory or specific marker files.

    Args:
        start_path: Directory to start searching from (defaults to current working directory)
        git_marker: Whether to look for .git directory (default: True)
        custom_markers: List of tuples (file/dir, file/dir, ...) that indicate repo root
                       Each tuple represents a combination that must all exist

    Returns:
        Path to repository root, or None if not found

    Examples:
        >>> find_repo_root()  # Look for .git from current directory
        PosixPath('/path/to/repo')

        >>> find_repo_root(custom_markers=[('requirements.txt', 'devops')])
        PosixPath('/path/to/repo')

        >>> find_repo_root(custom_markers=[('package.json',), ('Cargo.toml',)])
        PosixPath('/path/to/repo')  # Finds first match
    """
    current = Path(start_path) if start_path else Path.cwd()
    current = current.resolve()  # Resolve symlinks

    search_paths = [current] + list(current.parents)

    # Look for .git directory first (most reliable)
    if git_marker:
        for parent in search_paths:
            if (parent / ".git").exists():
                return parent

    # Look for custom marker combinations
    if custom_markers:
        for parent in search_paths:
            for marker_combo in custom_markers:
                if all((parent / marker).exists() for marker in marker_combo):
                    return parent

    return None


def find_metta_repo_root(start_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Find the Metta repository root specifically.

    This is a convenience function that looks for Metta-specific markers.
    """
    return find_repo_root(
        start_path=start_path,
        custom_markers=[
            ("requirements.txt", "devops"),  # Metta-specific combination
            ("metta",),  # Metta package directory
        ],
    )


def ensure_repo_root(target_root: Optional[Path] = None, auto_change: bool = True, quiet: bool = False) -> Path:
    """
    Ensure we're running from the repository root, optionally changing to it.

    Args:
        target_root: Specific repository root to use (if None, will search for it)
        auto_change: Whether to automatically change directories (default: True)
        quiet: Whether to suppress output messages (default: False)

    Returns:
        Path to repository root

    Raises:
        SystemExit: If repository root cannot be found
    """
    if target_root is None:
        repo_root = find_metta_repo_root()
    else:
        repo_root = Path(target_root).resolve()

    current_dir = Path.cwd().resolve()

    if repo_root is None:
        if not quiet:
            print(red("Error: Cannot find repository root. Please run this script from within the repository."))
            print(yellow("Looking for .git directory or requirements.txt + devops/ folder."))
        sys.exit(1)

    if current_dir != repo_root:
        if auto_change:
            if not quiet:
                print(yellow(f"Changing directory from {current_dir} to repository root: {repo_root}"))
            os.chdir(repo_root)
            if not quiet:
                print(green(f"Now running from: {Path.cwd()}"))
        else:
            if not quiet:
                print(yellow(f"Note: Running from {current_dir}, repository root is {repo_root}"))

    return repo_root


def ensure_metta_repo_root(auto_change: bool = True, quiet: bool = False) -> Path:
    """
    Ensure we're running from the Metta repository root.

    Convenience function for Metta-specific usage.
    """
    return ensure_repo_root(auto_change=auto_change, quiet=quiet)


def find_project_file(
    filename: str, start_path: Optional[Union[str, Path]] = None, stop_at_repo_root: bool = True
) -> Optional[Path]:
    """
    Find a project file by walking up the directory tree.

    Args:
        filename: Name of file to find (e.g., 'pyproject.toml', 'package.json')
        start_path: Directory to start searching from
        stop_at_repo_root: Whether to stop searching at repository boundary

    Returns:
        Path to the found file, or None if not found
    """
    current = Path(start_path) if start_path else Path.cwd()
    current = current.resolve()

    repo_root = find_repo_root(current) if stop_at_repo_root else None
    search_paths = [current] + list(current.parents)

    for parent in search_paths:
        target_file = parent / filename
        if target_file.exists():
            return target_file

        # Stop at repo root if requested
        if stop_at_repo_root and repo_root and parent == repo_root:
            break

    return None


def is_under_repo(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Check if the given path (or current directory) is under a repository.

    Args:
        path: Path to check (defaults to current working directory)

    Returns:
        True if under a repository, False otherwise
    """
    return find_repo_root(path) is not None


def get_relative_to_repo(path: Optional[Union[str, Path]] = None, repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Get path relative to repository root.

    Args:
        path: Path to make relative (defaults to current working directory)
        repo_root: Repository root to use (if None, will search for it)

    Returns:
        Path relative to repository root, or None if not under a repository
    """
    if repo_root is None:
        repo_root = find_repo_root()

    if repo_root is None:
        return None

    target_path = Path(path) if path else Path.cwd()
    target_path = target_path.resolve()
    repo_root = repo_root.resolve()

    try:
        return target_path.relative_to(repo_root)
    except ValueError:
        # Path is not under repo_root
        return None
