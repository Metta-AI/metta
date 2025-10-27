"""Tests for git repository filtering functionality."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from gitta import filter_repo


def create_repo_with_files():
    """Create a temporary git repository with multiple files."""
    tmpdir = tempfile.mkdtemp()
    repo_path = Path(tmpdir)

    # Initialize repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create directory structure
    (repo_path / "src").mkdir()
    (repo_path / "src" / "main.py").write_text("# Main file")
    (repo_path / "src" / "utils.py").write_text("# Utils")

    (repo_path / "docs").mkdir()
    (repo_path / "docs" / "README.md").write_text("# Documentation")

    (repo_path / "tests").mkdir()
    (repo_path / "tests" / "test_main.py").write_text("# Tests")

    (repo_path / "LICENSE").write_text("MIT License")
    (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/")

    # Commit all files
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Add more commits
    (repo_path / "src" / "feature.py").write_text("# New feature")
    subprocess.run(["git", "add", "src/feature.py"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add feature"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


def test_filter_repo_not_a_git_repository():
    """Test error when path is not a git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Not a git repository"):
            filter_repo(Path(tmpdir), ["src/"], make_root="src/")


def test_filter_repo_requires_make_root():
    """Test that filter_repo requires make_root parameter."""
    repo_path = create_repo_with_files()

    with pytest.raises(ValueError, match="requires make_root parameter"):
        filter_repo(repo_path, ["src/"])


def test_filter_repo_single_path_only():
    """Test that filter_repo only supports a single path."""
    repo_path = create_repo_with_files()

    with pytest.raises(ValueError, match="only supports a single path"):
        filter_repo(repo_path, ["src/", "docs/"], make_root="src/")


def test_filter_repo_nonexistent_path():
    """Test error when path doesn't exist in repository."""
    repo_path = create_repo_with_files()

    with pytest.raises(ValueError, match="does not exist"):
        filter_repo(repo_path, ["nonexistent/"], make_root="nonexistent/")


def test_filter_repo_basic():
    """Test basic filtering functionality using git subtree split."""
    repo_path = create_repo_with_files()

    # Filter to just the src directory, making it the root
    result_path = filter_repo(repo_path, ["src/"], make_root="src/")

    assert result_path.exists()
    assert (result_path / ".git").exists()

    # Check that only src files remain (and they're at the root)
    files = list(result_path.rglob("*"))
    file_names = [f.name for f in files if f.is_file()]

    # Should have src files at root level
    assert "main.py" in file_names
    assert "utils.py" in file_names
    assert "feature.py" in file_names

    # Should NOT have other files
    assert "README.md" not in file_names
    assert "test_main.py" not in file_names
    assert "LICENSE" not in file_names

    # Files should be at root, not in src/ subdirectory
    assert (result_path / "main.py").exists()
    assert (result_path / "utils.py").exists()
    assert (result_path / "feature.py").exists()
    assert not (result_path / "src").exists()


def test_filter_repo_preserves_history():
    """Test that commit history is preserved when filtering."""
    repo_path = create_repo_with_files()

    # Filter to just the src directory
    result_path = filter_repo(repo_path, ["src/"], make_root="src/")

    # Check that we have commits in the filtered repo
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=result_path,
        capture_output=True,
        text=True,
        check=True,
    )

    log_output = result.stdout
    # Should have the "Add feature" commit
    assert "Add feature" in log_output
    # Should have the initial commit
    assert "Initial commit" in log_output
