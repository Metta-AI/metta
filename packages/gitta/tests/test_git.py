"""Tests for git operations using real git commands in temporary directories."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from gitta import (
    GitError,
    add_remote,
    canonical_remote_url,
    find_root,
    get_all_remotes,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_remote_url,
    has_unstaged_changes,
    https_remote_url,
    validate_commit_state,
)


def create_temp_repo():
    """Create a temporary git repository and return its path."""
    tmpdir = tempfile.mkdtemp()
    repo_path = Path(tmpdir)

    # Initialize repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)

    # Create initial commit
    (repo_path / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

    return repo_path


def test_basic_git_operations():
    """Test basic git operations in a real repository."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Get current branch
    branch = get_current_branch()
    assert branch in ["main", "master"]

    # Get current commit
    commit = get_current_commit()
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)

    # List files
    files = get_file_list()
    assert "README.md" in files


def test_working_tree_changes():
    """Test detection of working tree changes."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Clean repo should have no changes
    has_changes, status = has_unstaged_changes()
    assert has_changes is False
    assert status == ""

    # Add a new file
    (repo_path / "new.txt").write_text("new content")
    has_changes, status = has_unstaged_changes()
    assert has_changes is True
    assert "new.txt" in status

    # Stage the file
    subprocess.run(["git", "add", "new.txt"], cwd=repo_path, check=True, capture_output=True)
    has_changes, status = has_unstaged_changes()
    assert has_changes is True  # Staged changes still count

    # Commit the file
    subprocess.run(["git", "commit", "-m", "Add new file"], cwd=repo_path, check=True, capture_output=True)
    has_changes, status = has_unstaged_changes()
    assert has_changes is False


def test_remote_operations():
    """Test git remote operations."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # No remotes initially
    assert get_remote_url() is None
    assert get_all_remotes() == {}

    # Add a remote
    add_remote("origin", "git@github.com:test/repo.git")
    assert get_remote_url() == "git@github.com:test/repo.git"

    # Add another remote
    add_remote("upstream", "https://github.com/upstream/repo.git")

    # List all remotes
    remotes = get_all_remotes()
    assert len(remotes) == 2
    assert remotes["origin"] == "git@github.com:test/repo.git"
    assert remotes["upstream"] in ("https://github.com/upstream/repo.git", "git@github.com:upstream/repo.git")


def test_find_git_root():
    """Test finding git repository root."""
    repo_path = create_temp_repo()

    # Create nested directories
    nested_dir = repo_path / "src" / "nested" / "deep"
    nested_dir.mkdir(parents=True)

    # Should find root from any subdirectory
    root = find_root(nested_dir)
    assert root and root.resolve() == repo_path.resolve()

    # Should return None outside a repo
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_root(Path(tmpdir)) is None


def test_https_remote_url_alias():
    """Test URL canonicalization helpers."""
    # GitHub SSH URLs
    assert https_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"
    assert https_remote_url("ssh://git@github.com/Owner/repo") == "https://github.com/Owner/repo"

    # GitHub HTTPS URLs
    assert https_remote_url("https://github.com/Owner/repo.git") == "https://github.com/Owner/repo"
    assert https_remote_url("https://github.com/Owner/repo") == "https://github.com/Owner/repo"

    # Non-GitHub URLs remain unchanged
    assert https_remote_url("git@gitlab.com:owner/repo.git") == "git@gitlab.com:owner/repo.git"

    # Backwards compatibility alias still available
    assert canonical_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"


def test_git_errors():
    """Test error handling outside a git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Operations should fail gracefully
        with pytest.raises(GitError):
            get_current_branch()

        with pytest.raises(GitError):
            get_current_commit()

        with pytest.raises(GitError):
            has_unstaged_changes()


def test_validate_commit_state_clean_repo():
    """Test validate_commit_state with a clean repository."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Clean repo should validate successfully (without requiring pushed)
    commit = validate_commit_state(require_clean=True, require_pushed=False)
    assert len(commit) == 40
    assert commit == get_current_commit()


def test_validate_commit_state_uncommitted_changes():
    """Test validate_commit_state fails with uncommitted changes."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Modify a file
    (repo_path / "README.md").write_text("# Modified")

    # Should fail when requiring clean
    with pytest.raises(GitError, match="uncommitted changes"):
        validate_commit_state(require_clean=True, require_pushed=False)

    # Should succeed when not requiring clean
    commit = validate_commit_state(require_clean=False, require_pushed=False)
    assert len(commit) == 40


def test_validate_commit_state_untracked_files():
    """Test validate_commit_state with untracked files."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Add an untracked file
    (repo_path / "new_file.txt").write_text("new content")

    # Should fail by default (untracked files count as changes)
    with pytest.raises(GitError, match="uncommitted changes"):
        validate_commit_state(require_clean=True, require_pushed=False)

    # Should succeed when allowing untracked files
    commit = validate_commit_state(require_clean=True, require_pushed=False, allow_untracked=True)
    assert len(commit) == 40


def test_validate_commit_state_unpushed_commit():
    """Test validate_commit_state with unpushed commits."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Set up a remote
    add_remote("origin", "git@github.com:test/repo.git")

    # Should fail when requiring pushed (no upstream configured)
    with pytest.raises(GitError, match="hasn't been pushed"):
        validate_commit_state(require_clean=True, require_pushed=True)

    # Should succeed when not requiring pushed
    commit = validate_commit_state(require_clean=True, require_pushed=False)
    assert len(commit) == 40


def test_validate_commit_state_wrong_repo():
    """Test validate_commit_state with wrong repository."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Add a remote
    add_remote("origin", "git@github.com:test/repo.git")

    # Should succeed when target_repo matches
    commit = validate_commit_state(require_clean=True, require_pushed=False, target_repo="test/repo")
    assert len(commit) == 40

    # Should fail when target_repo doesn't match
    with pytest.raises(GitError, match="Not in repository"):
        validate_commit_state(require_clean=True, require_pushed=False, target_repo="different/repo")
