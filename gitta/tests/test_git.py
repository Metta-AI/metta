"""Tests for git operations and utilities."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gitta import (
    GitError,
    add_remote,
    canonical_remote_url,
    find_root,
    get_all_remotes,
    get_branch_commit,
    get_commit_count,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_remote_url,
    has_unstaged_changes,
    is_commit_pushed,
    is_repo_match,
    validate_git_ref,
)


@pytest.fixture
def temp_repo():
    """Create a real temporary git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)

        # Create initial commit
        (repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

        yield repo_path


@pytest.fixture
def repo_with_changes(temp_repo):
    """Create a repo with some uncommitted changes."""
    (temp_repo / "modified.txt").write_text("modified content")
    subprocess.run(["git", "add", "modified.txt"], cwd=temp_repo, check=True, capture_output=True)
    (temp_repo / "README.md").write_text("# Modified README")
    (temp_repo / "untracked.txt").write_text("untracked content")
    return temp_repo


class TestRepositoryState:
    """Test repository state detection."""

    def test_clean_working_tree(self, temp_repo):
        """Test detection of clean working tree."""
        os.chdir(temp_repo)

        has_changes, status = has_unstaged_changes()
        assert has_changes is False
        assert status == ""

    def test_detect_staged_changes(self, repo_with_changes):
        """Test detection of staged changes."""
        os.chdir(repo_with_changes)

        has_changes, status = has_unstaged_changes()
        assert has_changes is True
        assert "modified.txt" in status
        assert "README.md" in status

    def test_detect_untracked_files(self, repo_with_changes):
        """Test detection of untracked files."""
        os.chdir(repo_with_changes)

        # Should detect untracked by default
        has_changes, status = has_unstaged_changes()
        assert has_changes is True
        assert "untracked.txt" in status

        # Can ignore untracked
        has_changes, status = has_unstaged_changes(allow_untracked=True)
        assert has_changes is True  # Still has staged changes

    def test_get_file_list(self, temp_repo):
        """Test listing files in a repository."""
        os.chdir(temp_repo)

        files = get_file_list()
        assert "README.md" in files
        assert len(files) >= 1

    def test_get_commit_count(self, temp_repo):
        """Test counting commits."""
        os.chdir(temp_repo)

        count = get_commit_count()
        assert count == 1  # Just the initial commit

        # Add another commit
        (temp_repo / "new.txt").write_text("new file")
        subprocess.run(["git", "add", "new.txt"], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Second commit"], cwd=temp_repo, check=True, capture_output=True)

        count = get_commit_count()
        assert count == 2

    def test_get_commit_message(self, temp_repo):
        """Test getting commit messages."""
        os.chdir(temp_repo)

        msg = get_commit_message("HEAD")
        assert msg == "Initial commit"

        # Add a multiline commit
        (temp_repo / "test.txt").write_text("test")
        subprocess.run(["git", "add", "test.txt"], cwd=temp_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Title\n\nBody text"], cwd=temp_repo, check=True, capture_output=True)

        msg = get_commit_message("HEAD")
        assert "Title" in msg
        assert "Body text" in msg


class TestRemoteOperations:
    """Test remote operations."""

    def test_add_and_list_remotes(self, temp_repo):
        """Test adding and listing remotes."""
        os.chdir(temp_repo)

        # Initially no remotes
        assert get_remote_url() is None
        assert get_all_remotes() == {}

        # Add origin
        add_remote("origin", "git@github.com:test/repo.git")
        assert get_remote_url() == "git@github.com:test/repo.git"
        assert get_remote_url("origin") == "git@github.com:test/repo.git"

        # Add upstream
        add_remote("upstream", "git@github.com:upstream/repo.git")
        assert get_remote_url("upstream") == "git@github.com:upstream/repo.git"

        # List all remotes
        remotes = get_all_remotes()
        assert len(remotes) == 2
        assert remotes["origin"] == "git@github.com:test/repo.git"
        assert remotes["upstream"] == "git@github.com:upstream/repo.git"

    def test_remote_url_canonicalization(self):
        """Test URL canonicalization without a repo."""
        # SSH formats
        assert canonical_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"
        assert canonical_remote_url("ssh://git@github.com/Owner/repo") == "https://github.com/Owner/repo"

        # HTTPS formats
        assert canonical_remote_url("https://github.com/Owner/repo.git") == "https://github.com/Owner/repo"
        assert canonical_remote_url("https://github.com/Owner/repo") == "https://github.com/Owner/repo"

        # Non-GitHub URLs unchanged
        assert canonical_remote_url("git@gitlab.com:owner/repo.git") == "git@gitlab.com:owner/repo.git"
        assert canonical_remote_url("https://bitbucket.org/owner/repo") == "https://bitbucket.org/owner/repo"

    @patch("gitta.git.get_all_remotes")
    def test_detect_repo_match(self, mock_get_all_remotes):
        """Test detection of repo match with various URL formats."""
        test_cases = [
            {"origin": "git@github.com:Metta-AI/metta.git"},
            {"origin": "https://github.com/Metta-AI/metta.git"},
            {"upstream": "git@github.com:Metta-AI/metta"},  # Different remote
            {"fork": "https://github.com/Metta-AI/metta"},  # No .git
        ]

        for remotes in test_cases:
            mock_get_all_remotes.return_value = remotes
            assert is_repo_match("Metta-AI/metta") is True

        # Non-matching repos
        mock_get_all_remotes.return_value = {"origin": "git@github.com:other/repo.git"}
        assert is_repo_match("Metta-AI/metta") is False

        mock_get_all_remotes.return_value = {}
        assert is_repo_match("Metta-AI/metta") is False


class TestBranchOperations:
    """Test branch and commit operations."""

    def test_validate_refs(self, temp_repo):
        """Test reference validation in a real repository."""
        os.chdir(temp_repo)

        # HEAD should always be valid
        commit = validate_git_ref("HEAD")
        assert commit is not None
        assert len(commit) == 40

        # Current branch should be valid
        branch = get_current_branch()
        assert validate_git_ref(branch) is not None

        # Invalid refs should return None
        assert validate_git_ref("definitely-not-a-branch") is None
        assert validate_git_ref("") is None
        assert validate_git_ref("invalid..ref") is None

    def test_get_branch_commit(self, temp_repo):
        """Test getting branch commits."""
        os.chdir(temp_repo)

        branch = get_current_branch()
        commit = get_branch_commit(branch)
        assert len(commit) == 40

        # Should be same as current commit
        assert commit == get_current_commit()

    def test_detached_head_fallback(self, temp_repo):
        """Test behavior in detached HEAD state."""
        os.chdir(temp_repo)

        # Get current commit
        commit = get_current_commit()

        # Checkout commit directly (detached HEAD)
        subprocess.run(["git", "checkout", commit], cwd=temp_repo, capture_output=True)

        # Should return commit hash as branch name
        branch = get_current_branch()
        assert branch == commit


class TestPushStatus:
    """Test push status detection."""

    @patch("gitta.git.run_git")
    def test_is_commit_pushed_with_tracking(self, mock_run_git):
        """Test checking if commit is pushed when tracking branch exists."""
        # Mock that commit is in remote branch
        mock_run_git.side_effect = [
            "abc123",  # First: verify commit exists
            "main",  # Second: get current branch
            "origin/main",  # Third: get remote tracking branch
            None,  # Fourth: merge-base succeeds (no exception)
        ]
        assert is_commit_pushed("abc123") is True

        # Mock that commit is not in remote branch
        mock_run_git.side_effect = [
            "abc123",  # First: verify commit exists
            "main",  # Second: get current branch
            "origin/main",  # Third: get remote tracking branch
            GitError("not ancestor"),  # Fourth: merge-base fails
        ]
        assert is_commit_pushed("abc123") is False

    @patch("gitta.git.run_git")
    def test_is_commit_pushed_without_tracking(self, mock_run_git):
        """Test checking if commit is pushed when no tracking branch."""
        # Simulate error (no remote tracking) - falls back to checking remote branches
        mock_run_git.side_effect = [
            "abc123",  # First: verify commit exists
            "main",  # Second: get current branch
            GitError("no tracking branch"),  # Third: no remote tracking
            "  origin/feature\n  origin/main",  # Fourth: remote branches containing commit
        ]
        assert is_commit_pushed("abc123") is True

    @patch("gitta.git.run_git")
    def test_is_commit_pushed_invalid_commit(self, mock_run_git):
        """Test checking push status with invalid commit."""
        mock_run_git.side_effect = GitError("Invalid commit")

        with pytest.raises(GitError) as exc_info:
            is_commit_pushed("invalid")
        assert "Invalid commit hash" in str(exc_info.value)


class TestRepositoryNavigation:
    """Test repository navigation functions."""

    def test_find_root_from_subdirectory(self, temp_repo):
        """Test finding repository root from nested directory."""
        subdir = temp_repo / "src" / "nested" / "deep"
        subdir.mkdir(parents=True)

        root = find_root(subdir)
        assert root is not None
        assert root.resolve() == temp_repo.resolve()

    def test_find_root_from_file(self, temp_repo):
        """Test finding repository root from a file path."""
        file_path = temp_repo / "src" / "test.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("# test")

        root = find_root(file_path)
        assert root is not None
        assert root.resolve() == temp_repo.resolve()

    def test_find_root_outside_repo(self):
        """Test that find_root returns None outside a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_root(Path(tmpdir)) is None
