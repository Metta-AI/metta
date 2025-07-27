import json
import subprocess


class GitError(Exception):
    """Custom exception for git-related errors."""


METTA_API_REPO_URL = "https://github.com/Metta-AI/metta.git"


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("Git is not installed!") from e


def run_gh(*args: str) -> str:
    """Run a GitHub CLI command and return its output."""
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"GitHub CLI command failed ({e.returncode}): {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise GitError("GitHub CLI (gh) is not installed!") from e


def get_current_branch() -> str:
    """Get the current git branch name."""
    try:
        return run_git("symbolic-ref", "--short", "HEAD")
    except GitError as e:
        if "not a git repository" in str(e):
            raise ValueError("Not in a git repository") from e
        elif "HEAD is not a symbolic ref" in str(e):
            return get_current_commit()
        raise


def get_current_commit() -> str:
    """Get the current git commit hash."""
    return run_git("rev-parse", "HEAD")


def get_branch_commit(branch: str) -> str:
    """Get the commit hash for a given branch."""
    # Fetch quietly to ensure we have latest remote data
    try:
        run_git("fetch", "--quiet")
    except GitError:
        # Fetch failure is non-fatal, continue with local data
        pass

    return run_git("rev-parse", branch)


def get_commit_message(commit_hash: str) -> str:
    """Get the commit message for a given commit."""
    return run_git("log", "-1", "--pretty=%B", commit_hash)


def has_unstaged_changes() -> bool:
    """Check if there are any unstaged changes."""
    status_output = run_git("status", "--porcelain")
    return bool(status_output)


def is_commit_pushed(commit_hash: str) -> bool:
    """Check if a commit has been pushed to any remote branch."""

    # Get all remote branches that contain this commit
    remote_branches = run_git("branch", "-r", "--contains", commit_hash)
    return bool(remote_branches.strip())


def validate_git_ref(ref: str) -> str | None:
    """Validate a git reference exists (locally or in remote)."""
    try:
        commit_hash = run_git("rev-parse", "--verify", ref)
    except GitError:
        return None
    return commit_hash


def get_matched_pr(commit_hash: str) -> tuple[int, str] | None:
    """
    Check if a commit is the HEAD of an open PR.

    Returns:
        tuple(pr_number, pr_title) if commit is HEAD of an open PR, None otherwise
    """

    # Get ALL open PRs by setting a high limit
    pr_json = run_gh("pr", "list", "--state", "open", "--limit", "999", "--json", "number,title,headRefOid")
    prs = json.loads(pr_json)

    for pr in prs:
        # Check if this PR's HEAD commit matches our commit
        pr_head_sha = pr.get("headRefOid", "")

        # Compare commits (handle both short and full hashes)
        if pr_head_sha.startswith(commit_hash) or commit_hash.startswith(pr_head_sha):
            return (int(pr["number"]), pr["title"])

    return None
