"""Git utilities library for Metta projects."""

# Import all functions from git.py
from gitlib.git import (
    METTA_API_REPO,
    METTA_API_REPO_URL,
    METTA_GITHUB_ORGANIZATION,
    METTA_GITHUB_REPO,
    GitError,
    add_remote,
    diff,
    fetch,
    find_root,
    get_branch_commit,
    get_commit_count,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_git_hash_for_remote_task,
    get_latest_commit,
    get_matched_pr,
    get_remote_url,
    has_unstaged_changes,
    is_commit_pushed,
    is_metta_ai_repo,
    memoize,
    ref_exists,
    run_gh,
    run_git,
    run_git_in_dir,
    run_git_with_cwd,
    validate_git_ref,
)

# Import filter_repo from git_filter.py
from gitlib.git_filter import filter_repo

# Import post_commit_status from github.py
from gitlib.github import post_commit_status

__all__ = [
    # Git functions
    "GitError",
    "METTA_API_REPO",
    "METTA_API_REPO_URL",
    "METTA_GITHUB_ORGANIZATION",
    "METTA_GITHUB_REPO",
    "run_git_with_cwd",
    "run_git",
    "run_git_in_dir",
    "run_gh",
    "get_current_branch",
    "get_current_commit",
    "get_branch_commit",
    "get_commit_message",
    "has_unstaged_changes",
    "is_commit_pushed",
    "validate_git_ref",
    "get_matched_pr",
    "get_remote_url",
    "is_metta_ai_repo",
    "get_git_hash_for_remote_task",
    "get_latest_commit",
    "get_file_list",
    "get_commit_count",
    "add_remote",
    "find_root",
    "fetch",
    "ref_exists",
    "diff",
    "memoize",
    # GitHub functions
    "post_commit_status",
    # Git filter functions
    "filter_repo",
]
