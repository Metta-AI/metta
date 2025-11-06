"""Tests for git operations using real git commands in temporary directories."""

import os
import pathlib
import subprocess
import tempfile

import pytest

import gitta


def create_temp_repo():
    """Create a temporary git repository and return its path."""
    tmpdir = tempfile.mkdtemp()
    repo_path = pathlib.Path(tmpdir)

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
    branch = gitta.get_current_branch()
    assert branch in ["main", "master"]

    # Get current commit
    commit = gitta.get_current_commit()
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)

    # List files
    files = gitta.get_file_list()
    assert "README.md" in files


def test_working_tree_changes():
    """Test detection of working tree changes."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Clean repo should have no changes
    has_changes, status = gitta.has_unstaged_changes()
    assert has_changes is False
    assert status == ""

    # Add a new file
    (repo_path / "new.txt").write_text("new content")
    has_changes, status = gitta.has_unstaged_changes()
    assert has_changes is True
    assert "new.txt" in status

    # Stage the file
    subprocess.run(["git", "add", "new.txt"], cwd=repo_path, check=True, capture_output=True)
    has_changes, status = gitta.has_unstaged_changes()
    assert has_changes is True  # Staged changes still count

    # Commit the file
    subprocess.run(["git", "commit", "-m", "Add new file"], cwd=repo_path, check=True, capture_output=True)
    has_changes, status = gitta.has_unstaged_changes()
    assert has_changes is False


def test_remote_operations():
    """Test git remote operations."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # No remotes initially
    assert gitta.get_remote_url() is None
    assert gitta.get_all_remotes() == {}

    # Add a remote
    gitta.add_remote("origin", "git@github.com:test/repo.git")
    assert gitta.get_remote_url() == "git@github.com:test/repo.git"

    # Add another remote
    gitta.add_remote("upstream", "https://github.com/upstream/repo.git")

    # List all remotes
    remotes = gitta.get_all_remotes()
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
    root = gitta.find_root(nested_dir)
    assert root and root.resolve() == repo_path.resolve()

    # Should return None outside a repo
    with tempfile.TemporaryDirectory() as tmpdir:
        assert gitta.find_root(pathlib.Path(tmpdir)) is None


def test_https_remote_url_alias():
    """Test URL canonicalization helpers."""
    # GitHub SSH URLs
    assert gitta.https_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"
    assert gitta.https_remote_url("ssh://git@github.com/Owner/repo") == "https://github.com/Owner/repo"

    # GitHub HTTPS URLs
    assert gitta.https_remote_url("https://github.com/Owner/repo.git") == "https://github.com/Owner/repo"
    assert gitta.https_remote_url("https://github.com/Owner/repo") == "https://github.com/Owner/repo"

    # Non-GitHub URLs remain unchanged
    assert gitta.https_remote_url("git@gitlab.com:owner/repo.git") == "git@gitlab.com:owner/repo.git"

    # Backwards compatibility alias still available
    assert gitta.canonical_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"


def test_git_errors():
    """Test error handling outside a git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Operations should fail gracefully
        with pytest.raises(gitta.GitError):
            gitta.get_current_branch()

        with pytest.raises(gitta.GitError):
            gitta.get_current_commit()

        with pytest.raises(gitta.GitError):
            gitta.has_unstaged_changes()


def test_validate_commit_state_clean_repo():
    """Test validate_commit_state with a clean repository."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Clean repo should validate successfully (without requiring pushed)
    commit = gitta.validate_commit_state(require_clean=True, require_pushed=False)
    assert len(commit) == 40
    assert commit == gitta.get_current_commit()


def test_validate_commit_state_uncommitted_changes():
    """Test validate_commit_state fails with uncommitted changes."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Modify a file
    (repo_path / "README.md").write_text("# Modified")

    # Should fail when requiring clean
    with pytest.raises(gitta.GitError, match="uncommitted changes"):
        gitta.validate_commit_state(require_clean=True, require_pushed=False)

    # Should succeed when not requiring clean
    commit = gitta.validate_commit_state(require_clean=False, require_pushed=False)
    assert len(commit) == 40


def test_validate_commit_state_untracked_files():
    """Test validate_commit_state with untracked files."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Add an untracked file
    (repo_path / "new_file.txt").write_text("new content")

    # Should fail by default (untracked files count as changes)
    with pytest.raises(gitta.GitError, match="uncommitted changes"):
        gitta.validate_commit_state(require_clean=True, require_pushed=False)

    # Should succeed when allowing untracked files
    commit = gitta.validate_commit_state(require_clean=True, require_pushed=False, allow_untracked=True)
    assert len(commit) == 40


def test_validate_commit_state_unpushed_commit():
    """Test validate_commit_state with unpushed commits."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Set up a remote
    gitta.add_remote("origin", "git@github.com:test/repo.git")

    # Should fail when requiring pushed (no upstream configured)
    with pytest.raises(gitta.GitError, match="hasn't been pushed"):
        gitta.validate_commit_state(require_clean=True, require_pushed=True)

    # Should succeed when not requiring pushed
    commit = gitta.validate_commit_state(require_clean=True, require_pushed=False)
    assert len(commit) == 40


def test_validate_commit_state_wrong_repo():
    """Test validate_commit_state with wrong repository."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Add a remote
    gitta.add_remote("origin", "git@github.com:test/repo.git")

    # Should succeed when target_repo matches
    commit = gitta.validate_commit_state(require_clean=True, require_pushed=False, target_repo="test/repo")
    assert len(commit) == 40

    # Should fail when target_repo doesn't match
    with pytest.raises(gitta.GitError, match="Not in repository"):
        gitta.validate_commit_state(require_clean=True, require_pushed=False, target_repo="different/repo")
