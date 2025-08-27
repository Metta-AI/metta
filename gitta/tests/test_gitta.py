"""
Behavioral tests for gitta - Git utilities for Metta projects.

These tests focus on real-world workflows and expected behaviors,
avoiding implementation details and monkeypatching where possible.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gitta import (
    GitError,
    GitNotInstalledError,
    add_remote,
    canonical_remote_url,
    find_root,
    get_all_remotes,
    get_branch_commit,
    get_commit_count,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_remote_url,
    has_unstaged_changes,
    is_commit_pushed,
    is_metta_ai_repo,
    run_git,
    run_git_in_dir,
    validate_git_ref,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        (repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

        yield repo_path


@pytest.fixture
def current_repo():
    """Use the actual gitta repository for tests that need real git history."""
    # Find the gitta repo root
    current_file = Path(__file__)
    repo_root = current_file.parent.parent.parent.parent  # Go up to metta root
    if not (repo_root / ".git").exists():
        pytest.skip("Not in a git repository")

    old_cwd = os.getcwd()
    os.chdir(repo_root)
    yield repo_root
    os.chdir(old_cwd)


# ============================================================================
# Basic Git Operations
# ============================================================================


class TestBasicGitOperations:
    """Test basic git command execution."""

    def test_run_git_in_real_repo(self, current_repo):
        """Test running git commands in the actual repository."""
        # Should work in a real repo
        branch = get_current_branch()
        assert isinstance(branch, str)
        assert len(branch) > 0

        commit = get_current_commit()
        assert isinstance(commit, str)
        assert len(commit) == 40  # Git SHA is 40 chars

    def test_run_git_in_temp_repo(self, temp_git_repo):
        """Test basic git operations in a temporary repo."""
        # Change to temp repo
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            # Should be on main or master branch
            branch = get_current_branch()
            assert branch in ["main", "master"]

            # Should have exactly one commit
            count = get_commit_count()
            assert count == 1

            # Should have one file
            files = get_file_list()
            assert "README.md" in files
        finally:
            os.chdir(old_cwd)

    def test_run_git_in_specific_directory(self, temp_git_repo):
        """Test running git commands with explicit directory."""
        # Should work from anywhere
        commit = run_git_in_dir(temp_git_repo, "rev-parse", "HEAD")
        assert len(commit) == 40

        # Verify it's the same as when run from inside
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)
        try:
            local_commit = get_current_commit()
            assert commit == local_commit
        finally:
            os.chdir(old_cwd)


# ============================================================================
# Repository Detection
# ============================================================================


class TestRepositoryDetection:
    """Test repository root finding and remote detection."""

    def test_find_repo_root(self, temp_git_repo):
        """Test finding repository root from various locations."""
        # From root itself
        assert find_root(temp_git_repo).resolve() == temp_git_repo.resolve()

        # From subdirectory
        subdir = temp_git_repo / "src" / "nested"
        subdir.mkdir(parents=True)
        assert find_root(subdir).resolve() == temp_git_repo.resolve()

        # From file
        file_path = subdir / "test.py"
        file_path.write_text("# test")
        assert find_root(file_path).resolve() == temp_git_repo.resolve()

    def test_find_root_outside_repo(self):
        """Test that find_root returns None outside a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_root(Path(tmpdir)) is None

    def test_detect_metta_repo(self):
        """Test detection of Metta AI repository."""
        # Mock the remotes to simulate Metta repo
        with patch("gitta.get_all_remotes") as mock_remotes:
            # Should detect various URL formats
            test_cases = [
                {"origin": "git@github.com:Metta-AI/metta.git"},
                {"origin": "https://github.com/Metta-AI/metta.git"},
                {"upstream": "git@github.com:Metta-AI/metta"},  # Different remote name
                {"fork": "https://github.com/Metta-AI/metta"},  # No .git suffix
            ]

            for remotes in test_cases:
                mock_remotes.return_value = remotes
                assert is_metta_ai_repo() is True

            # Should not detect other repos
            mock_remotes.return_value = {"origin": "git@github.com:other/repo.git"}
            assert is_metta_ai_repo() is False

            # Should handle no remotes
            mock_remotes.return_value = {}
            assert is_metta_ai_repo() is False


# ============================================================================
# Change Detection
# ============================================================================


class TestChangeDetection:
    """Test detecting uncommitted changes."""

    def test_clean_working_tree(self, temp_git_repo):
        """Test detection when working tree is clean."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            has_changes, status = has_unstaged_changes()
            assert has_changes is False
            assert status == ""
        finally:
            os.chdir(old_cwd)

    def test_unstaged_changes(self, temp_git_repo):
        """Test detection of unstaged changes."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            # Modify existing file
            (temp_git_repo / "README.md").write_text("# Modified")

            has_changes, status = has_unstaged_changes()
            assert has_changes is True
            assert "README.md" in status
        finally:
            os.chdir(old_cwd)

    def test_untracked_files(self, temp_git_repo):
        """Test detection of untracked files."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            # Add untracked file
            (temp_git_repo / "new.txt").write_text("new file")

            # Should detect untracked by default
            has_changes, status = has_unstaged_changes()
            assert has_changes is True
            assert "new.txt" in status

            # Can ignore untracked
            has_changes, status = has_unstaged_changes(allow_untracked=True)
            assert has_changes is False
        finally:
            os.chdir(old_cwd)


# ============================================================================
# Reference Validation
# ============================================================================


class TestReferenceValidation:
    """Test git reference validation and resolution."""

    def test_validate_common_refs(self, current_repo):
        """Test validation of common git references."""
        # HEAD should always be valid
        assert validate_git_ref("HEAD") is not None

        # Current branch should be valid
        branch = get_current_branch()
        if branch:  # Not in detached HEAD
            assert validate_git_ref(branch) is not None

    def test_validate_invalid_refs(self, current_repo):
        """Test that invalid references return None."""
        assert validate_git_ref("definitely-not-a-real-branch") is None
        assert validate_git_ref("") is None
        assert validate_git_ref("invalid..ref") is None

    def test_ref_returns_commit_hash(self, current_repo):
        """Test that validate_git_ref returns a commit hash."""
        commit = validate_git_ref("HEAD")
        assert commit is not None
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)


# ============================================================================
# Remote Operations
# ============================================================================


class TestRemoteOperations:
    """Test remote repository operations."""

    def test_add_and_get_remotes(self, temp_git_repo):
        """Test adding and retrieving remotes."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            # Initially no remotes
            assert get_remote_url() is None
            assert get_all_remotes() == {}

            # Add origin
            add_remote("origin", "git@github.com:test/repo.git")
            assert get_remote_url() == "git@github.com:test/repo.git"
            assert get_remote_url("origin") == "git@github.com:test/repo.git"

            # Add upstream (use SSH format as per user preference)
            add_remote("upstream", "git@github.com:upstream/repo.git")
            remotes = get_all_remotes()
            assert len(remotes) == 2
            assert remotes["origin"] == "git@github.com:test/repo.git"
            assert remotes["upstream"] == "git@github.com:upstream/repo.git"

            # Adding duplicate updates it (removes old, adds new)
            add_remote("origin", "git@github.com:other/repo.git")
            assert get_remote_url("origin") == "git@github.com:other/repo.git"
        finally:
            os.chdir(old_cwd)

    def test_canonical_remote_urls(self):
        """Test URL canonicalization for comparison."""
        # SSH formats
        assert canonical_remote_url("git@github.com:Owner/repo.git") == "https://github.com/Owner/repo"
        assert canonical_remote_url("git@github.com:Owner/repo") == "https://github.com/Owner/repo"
        assert canonical_remote_url("ssh://git@github.com/Owner/repo.git") == "https://github.com/Owner/repo"

        # HTTPS formats
        assert canonical_remote_url("https://github.com/Owner/repo.git") == "https://github.com/Owner/repo"
        assert canonical_remote_url("https://github.com/Owner/repo") == "https://github.com/Owner/repo"

        # Non-GitHub URLs unchanged
        assert canonical_remote_url("git@gitlab.com:owner/repo.git") == "git@gitlab.com:owner/repo.git"

        # Whitespace handling
        assert canonical_remote_url("  git@github.com:Owner/repo.git  ") == "https://github.com/Owner/repo"


# ============================================================================
# Push Status
# ============================================================================


class TestPushStatus:
    """Test checking if commits are pushed."""

    def test_unpushed_commit(self, temp_git_repo):
        """Test detection of unpushed commits."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            commit = get_current_commit()
            # No remote, so not pushed
            assert is_commit_pushed(commit) is False
        finally:
            os.chdir(old_cwd)

    def test_current_commit_in_real_repo(self, current_repo):
        """Test push status in real repository."""
        # Current commit in the actual repo should typically be pushed
        # (unless we're on a feature branch with local commits)
        commit = get_current_commit()

        # This is a behavioral test - we just verify it doesn't crash
        # and returns a boolean
        result = is_commit_pushed(commit)
        assert isinstance(result, bool)


# ============================================================================
# Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and reporting."""

    def test_git_not_installed(self):
        """Test helpful error when git is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(GitNotInstalledError) as exc:
                run_git("status")
            assert "not installed" in str(exc.value).lower()

    def test_not_in_repo_error(self):
        """Test error when not in a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(GitError):
                run_git_in_dir(tmpdir, "status")

    def test_command_failure(self):
        """Test handling of failed git commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, check=True)

            # Try to checkout non-existent branch
            with pytest.raises(GitError) as exc:
                run_git_in_dir(repo, "checkout", "non-existent-branch")
            assert "non-existent-branch" in str(exc.value)


# ============================================================================
# Branch and Commit Operations
# ============================================================================


class TestBranchOperations:
    """Test branch-related operations."""

    def test_get_branch_commit(self, current_repo):
        """Test getting commit for a branch."""
        branch = get_current_branch()
        if branch:
            commit = get_branch_commit(branch)
            assert len(commit) == 40

    def test_branch_in_detached_head(self, temp_git_repo):
        """Test behavior in detached HEAD state."""
        old_cwd = os.getcwd()
        os.chdir(temp_git_repo)

        try:
            # Get first commit
            first_commit = get_current_commit()

            # Create second commit
            (temp_git_repo / "file2.txt").write_text("content")
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Second commit"], check=True)

            # Checkout first commit (detached HEAD)
            subprocess.run(["git", "checkout", first_commit], check=True)

            # In detached HEAD, get_current_branch might return None or commit
            # This is a behavioral test - just verify it doesn't crash
            result = get_current_branch()
            assert result is None or len(result) == 40
        finally:
            os.chdir(old_cwd)
