# Create a new file: metta/util/github.py
import subprocess


def get_current_branch(repo_path=None):
    """Get the current git branch name."""
    try:
        cmd = ["git", "symbolic-ref", "--short", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "symbolic-ref", "--short", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_current_commit(repo_path=None):
    """Get the current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def is_commit_pushed(commit_hash, repo_path=None):
    """Check if a commit has been pushed to any remote branch."""
    try:
        cmd = ["git", "branch", "-r", "--contains", commit_hash]
        if repo_path:
            cmd = ["git", "-C", repo_path, "branch", "-r", "--contains", commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def has_unstaged_changes(repo_path=None):
    """Check if there are any unstaged changes in the git repository."""
    try:
        cmd = ["git", "status", "--porcelain"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "status", "--porcelain"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False
