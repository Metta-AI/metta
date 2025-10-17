"""Tests for core git command runner and error handling."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from gitta.core import (
    DubiousOwnershipError,
    GitError,
    GitNotInstalledError,
    NotAGitRepoError,
    run_git,
    run_git_cmd,
    run_git_in_dir,
)


def create_temp_repo():
    """Create a temporary git repository."""
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


def test_run_git_basic():
    """Test basic git command execution."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Test simple command
    output = run_git("status", "--short")
    assert output == ""  # Clean repo

    # Test command with output
    output = run_git("log", "--oneline", "-1")
    assert "Initial commit" in output


def test_run_git_in_dir():
    """Test running git commands with explicit directory."""
    repo_path = create_temp_repo()

    # Run from outside the repo
    with tempfile.TemporaryDirectory() as other_dir:
        os.chdir(other_dir)

        # Should work with explicit directory
        output = run_git_in_dir(repo_path, "status", "--short")
        assert output == ""

        # Get commit hash
        commit = run_git_in_dir(repo_path, "rev-parse", "HEAD")
        assert len(commit) == 40


def test_run_git_cmd_direct():
    """Test run_git_cmd function directly."""
    repo_path = create_temp_repo()

    # Test with various parameters
    output = run_git_cmd(["status"], cwd=repo_path, timeout=10.0)
    assert "working tree clean" in output or "nothing to commit" in output

    # Test with env overrides
    output = run_git_cmd(["config", "user.name"], cwd=repo_path, env_overrides={"GIT_CONFIG_GLOBAL": "/dev/null"})
    assert output == "Test User"


def test_not_a_git_repo_error():
    """Test error when running git outside a repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        with pytest.raises(NotAGitRepoError) as exc_info:
            run_git("status")

        assert "not in a git repository" in str(exc_info.value).lower()


def test_git_error_with_check_false():
    """Test that check=False returns empty string on error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run invalid command with check=False
        result = run_git_cmd(["invalid-command"], cwd=tmpdir, check=False)
        assert result == ""


def test_git_command_failure():
    """Test handling of general git command failures."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Try to checkout non-existent branch
    with pytest.raises(GitError) as exc_info:
        run_git("checkout", "non-existent-branch")

    assert "failed" in str(exc_info.value)
    assert "non-existent-branch" in str(exc_info.value)


def test_environment_variables():
    """Test that default environment variables are applied."""
    repo_path = create_temp_repo()

    # Add a file with changes to test pager
    (repo_path / "test.txt").write_text("line1\n" * 100)  # Many lines
    subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add test file"], cwd=repo_path, check=True, capture_output=True)

    # This would normally trigger a pager, but shouldn't with our env settings
    output = run_git_cmd(["log", "--oneline"], cwd=repo_path)
    lines = output.strip().split("\n")
    assert len(lines) == 2  # Two commits, all returned at once (no pager)


def test_output_encoding():
    """Test handling of different output encodings."""
    repo_path = create_temp_repo()

    # Create a file with unicode characters
    (repo_path / "unicode.txt").write_text("Hello 世界 🌍", encoding="utf-8")
    subprocess.run(["git", "add", "unicode.txt"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add unicode file"], cwd=repo_path, check=True, capture_output=True)

    # Should handle unicode in status output
    output = run_git_cmd(["log", "--oneline", "-1"], cwd=repo_path)
    assert "Add unicode file" in output


def test_timeout_handling():
    """Test command timeout handling (if we can simulate it)."""
    # This is tricky to test without mocking, but we can test that timeout parameter works
    repo_path = create_temp_repo()

    # Quick command should complete within timeout
    output = run_git_cmd(["status"], cwd=repo_path, timeout=5.0)
    assert "working tree clean" in output or "nothing to commit" in output


def test_git_not_installed_simulation():
    """Test error message format for git not installed (can't truly test without mocking)."""
    # We can't actually test this without mocking subprocess.run,
    # but we can verify the exception exists and has the right base class
    assert issubclass(GitNotInstalledError, GitError)

    # Test creating the exception
    exc = GitNotInstalledError("Test message")
    assert "Test message" in str(exc)


def test_dubious_ownership_simulation():
    """Test dubious ownership error format."""
    # Similarly, we can't trigger this without specific git setups,
    # but we can test the exception class
    assert issubclass(DubiousOwnershipError, GitError)

    # Test creating the exception with expected message format
    exc = DubiousOwnershipError(
        "fatal: detected dubious ownership\n\nTo fix this, run:\n  git config --global --add safe.directory /path"
    )
    assert "dubious ownership" in str(exc)
    assert "git config" in str(exc)


def test_error_with_stderr():
    """Test that stderr is properly captured in error messages."""
    repo_path = create_temp_repo()
    os.chdir(repo_path)

    # Force an error with stderr output
    with pytest.raises(GitError) as exc_info:
        run_git("log", "--invalid-option")

    # The error message should contain the stderr output
    error_msg = str(exc_info.value)
    assert "invalid" in error_msg.lower() or "unrecognized" in error_msg.lower()
