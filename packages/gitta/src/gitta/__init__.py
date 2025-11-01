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
    get_remote_url,
    git_log_since,
    has_unstaged_changes,
    https_remote_url,
    is_commit_pushed,
    is_repo_match,
    ref_exists,
    resolve_git_ref,
    validate_commit_state,
)

# GitHub API functionality
from .github import (
    create_pr,
    get_branches,
    get_commit_with_stats,
    get_commits,
    get_latest_commit,
    get_matched_pr,
    get_pull_requests,
    get_workflow_run_jobs,
    get_workflow_runs,
    github_client,
    list_all_workflow_runs,
    post_commit_status,
    run_gh,
)

# Secrets management (optional AWS support)
from .secrets import (
    clear_cache,
    get_anthropic_api_key,
    get_github_token,
    get_secret,
)

# PR splitting functionality
from .split import PRSplitter, split_pr

__all__ = [
    # Core
    "GitError",
    "GitNotInstalledError",
    "DubiousOwnershipError",
    "NotAGitRepoError",
    "run_git_cmd",
    "run_git",
    "run_git_in_dir",
    # Git operations
    "get_current_branch",
    "get_current_commit",
    "get_branch_commit",
    "get_commit_message",
    "has_unstaged_changes",
    "is_commit_pushed",
    "resolve_git_ref",
    "https_remote_url",
    "canonical_remote_url",  # Backwards compatibility alias
    "get_remote_url",
    "get_all_remotes",
    "is_repo_match",
    "get_file_list",
    "get_commit_count",
    "add_remote",
    "find_root",
    "fetch",
    "ref_exists",
    "diff",
    "git_log_since",
    "validate_commit_state",
    # GitHub API
    "run_gh",
    "get_matched_pr",
    "get_latest_commit",
    "post_commit_status",
    "create_pr",
    "github_client",
    "get_commits",
    "get_commit_with_stats",
    "get_workflow_runs",
    "get_workflow_run_jobs",
    "get_pull_requests",
    "get_branches",
    "list_all_workflow_runs",
    # Filter
    "filter_repo",
    # Secrets
    "get_secret",
    "get_github_token",
    "get_anthropic_api_key",
    "clear_cache",
    # Split
    "split_pr",
    "PRSplitter",
]
