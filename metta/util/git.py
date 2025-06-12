import subprocess
from typing import Optional


def get_current_branch(repo_path: Optional[str] = None) -> str:
    """Get the current git branch name."""
    cmd = ["git", "symbolic-ref", "--short", "HEAD"]
    if repo_path:
        cmd = ["git", "-C", repo_path, "symbolic-ref", "--short", "HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_current_commit(repo_path: Optional[str] = None) -> str:
    """Get the current git commit hash."""
    cmd = ["git", "rev-parse", "HEAD"]
    if repo_path:
        cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def is_commit_pushed(commit_hash: str, repo_path: Optional[str] = None) -> bool:
    """Check if a commit has been pushed to any remote branch."""
    cmd = ["git", "branch", "-r", "--contains", commit_hash]
    if repo_path:
        cmd = ["git", "-C", repo_path, "branch", "-r", "--contains", commit_hash]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def has_unstaged_changes(repo_path: Optional[str] = None) -> bool:
    """Check if there are any unstaged changes in the git repository."""
    cmd = ["git", "status", "--porcelain"]
    if repo_path:
        cmd = ["git", "-C", repo_path, "status", "--porcelain"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def get_branch_commit(branch_name: str, repo_path: Optional[str] = None) -> str:
    """Get the latest commit hash on a branch, including remote branches."""
    # Make sure we have the latest remote data
    fetch_cmd = ["git", "fetch", "--quiet"]
    if repo_path:
        fetch_cmd = ["git", "-C", repo_path, "fetch", "--quiet"]
    subprocess.run(fetch_cmd, check=True)

    # Get the commit hash for the branch
    rev_cmd = ["git", "rev-parse", branch_name]
    if repo_path:
        rev_cmd = ["git", "-C", repo_path, "rev-parse", branch_name]

    result = subprocess.run(rev_cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_commit_message(commit_hash: str, repo_path: Optional[str] = None) -> str:
    """Get the commit message for a specific commit hash."""
    cmd = ["git", "log", "-1", "--pretty=%B", commit_hash]
    if repo_path:
        cmd = ["git", "-C", repo_path, "log", "-1", "--pretty=%B", commit_hash]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
