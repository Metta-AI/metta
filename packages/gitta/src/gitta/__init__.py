"""Git utilities library."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

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
    get_uncommitted_files_and_statuses,
    git_log_since,
    has_uncommitted_changes,
    https_remote_url,
    is_commit_pushed,
    is_repo_match,
    ref_exists,
    resolve_git_ref,
    validate_commit_state,
)

if TYPE_CHECKING:
    from .github import (
        create_pr,
        get_commits,
        get_latest_commit,
        get_matched_pr,
        get_workflow_run_jobs,
        get_workflow_runs,
        github_client,
        post_commit_status,
        run_gh,
    )
    from .secrets import (
        clear_cache,
        get_anthropic_api_key,
        get_github_token,
        get_secret,
    )
    from .split import PRSplitter, split_pr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # GitHub API functionality
    "create_pr": ("gitta.github", "create_pr"),
    "get_commits": ("gitta.github", "get_commits"),
    "get_latest_commit": ("gitta.github", "get_latest_commit"),
    "get_matched_pr": ("gitta.github", "get_matched_pr"),
    "get_workflow_run_jobs": ("gitta.github", "get_workflow_run_jobs"),
    "get_workflow_runs": ("gitta.github", "get_workflow_runs"),
    "github_client": ("gitta.github", "github_client"),
    "post_commit_status": ("gitta.github", "post_commit_status"),
    "run_gh": ("gitta.github", "run_gh"),
    # Secrets management
    "clear_cache": ("gitta.secrets", "clear_cache"),
    "get_anthropic_api_key": ("gitta.secrets", "get_anthropic_api_key"),
    "get_github_token": ("gitta.secrets", "get_github_token"),
    "get_secret": ("gitta.secrets", "get_secret"),
    # PR splitting functionality
    "PRSplitter": ("gitta.split", "PRSplitter"),
    "split_pr": ("gitta.split", "split_pr"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'gitta' has no attribute '{name}'") from None
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()) | set(_LAZY_ATTRS.keys()))


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
    "has_uncommitted_changes",
    "get_uncommitted_files_and_statuses",
    "is_commit_pushed",
    "resolve_git_ref",
    "https_remote_url",
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
    "get_workflow_runs",
    "get_workflow_run_jobs",
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
