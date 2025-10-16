"""Git utilities library."""

from __future__ import annotations

# Core functionality
from .core import (
    DubiousOwnershipError,
    GitError,
    GitNotInstalledError,
    NotAGitRepoError,
    run_git,
    run_git_cmd,
    run_git_in_dir,
    run_git_with_cwd,
)

# Filter functionality
from .filter import filter_repo

# Git operations
from .git import (
    add_remote,
    canonical_remote_url,
    diff,
    fetch,
    find_root,
    get_all_remotes,
    get_branch_commit,
    get_commit_count,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_git_hash_for_remote_task,
    get_remote_url,
    has_unstaged_changes,
    is_commit_pushed,
    is_repo_match,
    ref_exists,
    validate_git_ref,
)

# GitHub API functionality
from .github import (
    create_pr,
    get_latest_commit,
    get_matched_pr,
    post_commit_status,
    run_gh,
)

__all__ = [
    # Core
    "GitError",
    "GitNotInstalledError",
    "DubiousOwnershipError",
    "NotAGitRepoError",
    "run_git_cmd",
    "run_git",
    "run_git_in_dir",
    "run_git_with_cwd",
    # Git operations
    "get_current_branch",
    "get_current_commit",
    "get_branch_commit",
    "get_commit_message",
    "has_unstaged_changes",
    "is_commit_pushed",
    "validate_git_ref",
    "canonical_remote_url",
    "get_remote_url",
    "get_all_remotes",
    "is_repo_match",
    "get_git_hash_for_remote_task",
    "get_file_list",
    "get_commit_count",
    "add_remote",
    "find_root",
    "fetch",
    "ref_exists",
    "diff",
    # GitHub API
    "run_gh",
    "get_matched_pr",
    "get_latest_commit",
    "post_commit_status",
    "create_pr",
    # Filter
    "filter_repo",
]
