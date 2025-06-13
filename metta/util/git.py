import subprocess


class GitError(Exception):
    """Custom exception for git-related errors."""


def run_git(*args: str) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if e.returncode == 129:
            raise GitError(f"Malformed git command or bad object: {e.stderr.strip()}") from e
        raise GitError(f"Git command failed ({e.returncode}): {e.stderr.strip()}") from e


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
    try:
        status_output = run_git("status", "--porcelain")
        return bool(status_output)
    except GitError:
        return False


def is_commit_pushed(commit_hash: str) -> bool:
    """Check if a commit has been pushed to any remote branch."""
    try:
        # Get all remote branches that contain this commit
        remote_branches = run_git("branch", "-r", "--contains", commit_hash)
        return bool(remote_branches.strip())
    except GitError:
        return False


def validate_git_ref(ref: str) -> str | None:
    """Validate a git reference exists (locally or in remote)."""
    try:
        commit_hash = run_git("rev-parse", "--verify", ref)
    except GitError:
        return None
    return commit_hash
