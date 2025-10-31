"""Shared utilities for linting workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from gitta import GitError, run_git_in_dir
from metta.setup.tools.code_formatters import partition_files_by_type
from metta.setup.utils import error, warning

SUPPORTED_LINT_TYPES: tuple[str, ...] = ("python", "json", "markdown", "shell", "toml", "yaml")


def parse_lint_type_option(raw: Optional[str]) -> list[str]:
    """Parse a comma-separated list of lint types."""
    if raw is None:
        return []

    entries = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not entries:
        return []

    if "all" in entries:
        return list(SUPPORTED_LINT_TYPES)

    invalid = [entry for entry in entries if entry not in SUPPORTED_LINT_TYPES]
    if invalid:
        raise ValueError(f"Unsupported format types: {', '.join(sorted(set(invalid)))}")

    ordered_unique: list[str] = []
    for entry in entries:
        if entry not in ordered_unique:
            ordered_unique.append(entry)
    return ordered_unique


def normalize_relative_paths(raw_paths: list[str], repo_root: Path) -> list[str]:
    """Return canonical repo-relative paths for linting."""
    normalized: list[str] = []
    for raw in raw_paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            warning(f"Skipping '{raw}': outside the repository at {repo_root}.")
            continue

        if not path.exists():
            warning(f"Skipping '{relative}': file does not exist.")
            continue

        if path.is_dir():
            warning(f"Skipping directory '{relative}'.")
            continue

        normalized.append(str(relative))
    return normalized


def get_tracked_files(repo_root: Path) -> list[str]:
    """Return the tracked files in the repository."""
    try:
        output = run_git_in_dir(repo_root, "ls-files")
    except GitError as git_error:
        error(f"Failed to list tracked files: {git_error}")
        raise typer.Exit(1) from git_error
    return [line.strip() for line in output.splitlines() if line.strip()]


def get_staged_files(repo_root: Path) -> list[str]:
    """Return the staged files (added/changed) in the repository."""
    try:
        output = run_git_in_dir(repo_root, "diff", "--cached", "--name-only", "--diff-filter=ACM")
    except GitError as git_error:
        error(f"Unable to inspect staged files: {git_error}")
        raise typer.Exit(1) from git_error
    return [line.strip() for line in output.splitlines() if line.strip()]


def select_files_by_type(files: list[str], types: Optional[list[str]]) -> list[str]:
    """Filter a sequence of files by detected lint types."""
    if not files:
        return []

    if not types:
        return sorted(dict.fromkeys(files))

    files_by_type = partition_files_by_type(files)
    selected: list[str] = []
    for entry in types:
        selected.extend(files_by_type.get(entry, []))
    return sorted(dict.fromkeys(selected))


def collect_files_for_types(repo_root: Path, types: list[str]) -> list[str]:
    """Return tracked files that match the requested types."""
    tracked = get_tracked_files(repo_root)
    return select_files_by_type(tracked, types)


def restage_modified_files(repo_root: Path, candidates: set[str]) -> list[str]:
    """Re-stage files modified during linting."""
    if not candidates:
        return []

    try:
        status_output = run_git_in_dir(repo_root, "status", "--porcelain")
    except GitError:
        return []

    to_stage: list[str] = []
    for raw in status_output.splitlines():
        if len(raw) < 4:
            continue
        path = raw[3:].strip()
        if path in candidates:
            to_stage.append(path)

    if to_stage:
        try:
            run_git_in_dir(repo_root, "add", *to_stage)
        except GitError as git_error:
            warning(f"Unable to re-stage files: {git_error}")

    return to_stage


__all__ = [
    "SUPPORTED_LINT_TYPES",
    "collect_files_for_types",
    "get_staged_files",
    "get_tracked_files",
    "normalize_relative_paths",
    "parse_lint_type_option",
    "restage_modified_files",
    "select_files_by_type",
]
