import subprocess


def get_current_branch() -> str:
    """Get the current git branch name."""
    result = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], capture_output=True, text=True, check=False)

    if result.returncode == 0:
        return result.stdout.strip()
    elif result.returncode == 128:
        # Not in a git repo or detached HEAD state
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")
        elif "HEAD is not a symbolic ref" in result.stderr:
            # In detached HEAD state - return commit hash instead
            return get_current_commit()

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def get_current_commit() -> str:
    """Get the current git commit hash."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)

    if result.returncode == 0:
        return result.stdout.strip()
    elif result.returncode == 128:
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")
        elif "ambiguous argument 'HEAD'" in result.stderr:
            raise ValueError("No commits in repository yet")

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def is_commit_pushed(commit_hash: str) -> bool:
    """Check if a commit has been pushed to any remote branch."""
    result = subprocess.run(
        ["git", "branch", "-r", "--contains", commit_hash], capture_output=True, text=True, check=False
    )

    if result.returncode == 0:
        return bool(result.stdout.strip())
    elif result.returncode == 129:
        # Invalid commit hash - treat as "not pushed"
        return False
    elif result.returncode == 128:
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def has_unstaged_changes() -> bool:
    """Check if there are any unstaged changes in the git repository."""
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)

    if result.returncode == 0:
        return bool(result.stdout.strip())
    elif result.returncode == 128:
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def get_branch_commit(branch_name: str) -> str:
    """Get the latest commit hash on a branch, including remote branches."""
    # Make sure we have the latest remote data
    fetch_result = subprocess.run(["git", "fetch", "--quiet"], capture_output=True, text=True, check=False)

    if fetch_result.returncode == 128:
        if "not a git repository" in fetch_result.stderr:
            raise ValueError("Not in a git repository")
    elif fetch_result.returncode != 0:
        raise subprocess.CalledProcessError(
            fetch_result.returncode, fetch_result.args, fetch_result.stdout, fetch_result.stderr
        )

    # Get the commit hash for the branch
    result = subprocess.run(["git", "rev-parse", branch_name], capture_output=True, text=True, check=False)

    if result.returncode == 0:
        return result.stdout.strip()
    elif result.returncode == 128:
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")
        elif "unknown revision" in result.stderr:
            raise ValueError(f"Branch '{branch_name}' not found")

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def get_commit_message(commit_hash: str) -> str:
    """Get the commit message for a specific commit hash."""
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B", commit_hash], capture_output=True, text=True, check=False
    )

    if result.returncode == 0:
        return result.stdout.strip()
    elif result.returncode == 128:
        if "not a git repository" in result.stderr:
            raise ValueError("Not in a git repository")
        elif "bad revision" in result.stderr or "unknown revision" in result.stderr:
            raise ValueError(f"Commit '{commit_hash}' not found")

    # Unexpected error
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
