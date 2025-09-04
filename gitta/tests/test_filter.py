"""Tests for git-filter-repo functionality without mocks."""

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
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)

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
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

    # Add more commits
    (repo_path / "src" / "feature.py").write_text("# New feature")
    subprocess.run(["git", "add", "src/feature.py"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add feature"], cwd=repo_path, check=True, capture_output=True)

    return repo_path


def test_filter_repo_not_a_git_repository():
    """Test error when path is not a git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError) as exc_info:
            filter_repo(Path(tmpdir), ["src"])
        assert "not a git repository" in str(exc_info.value).lower()


@pytest.mark.skipif(
    subprocess.run(["which", "git-filter-repo"], capture_output=True).returncode != 0,
    reason="git-filter-repo not installed",
)
def test_filter_repo_basic():
    """Test basic filtering functionality if git-filter-repo is installed."""
    repo_path = create_repo_with_files()

    # Filter to just the src directory
    result_path = filter_repo(repo_path, ["src/"])

    assert result_path.exists()
    assert (result_path / ".git").exists()

    # Check that only src files remain
    files = list(result_path.rglob("*"))
    file_names = [f.name for f in files if f.is_file()]

    # Should have src files
    assert "main.py" in file_names
    assert "utils.py" in file_names
    assert "feature.py" in file_names

    # Should NOT have other files
    assert "README.md" not in file_names
    assert "test_main.py" not in file_names
    assert "LICENSE" not in file_names


def test_filter_repo_tool_detection():
    """Test detection of git-filter-repo tool."""
    # This test just checks if we can detect whether the tool is installed
    result = subprocess.run(["which", "git-filter-repo"], capture_output=True)
    tool_installed = result.returncode == 0

    if not tool_installed:
        # If tool is not installed, we expect a RuntimeError
        repo_path = create_repo_with_files()

        with pytest.raises(RuntimeError) as exc_info:
            filter_repo(repo_path, ["src/"])

        assert "git-filter-repo not found" in str(exc_info.value)
        assert "metta install filter-repo" in str(exc_info.value)
