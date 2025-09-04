"""Tests for git-filter-repo functionality."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gitta import filter_repo


@pytest.fixture
def temp_repo_with_files():
    """Create a temporary git repository with multiple files and directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True
        )
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
        subprocess.run(
            ["git", "commit", "-m", "Initial commit with structure"], cwd=repo_path, check=True, capture_output=True
        )

        # Add more commits to test filtering
        (repo_path / "src" / "feature.py").write_text("# New feature")
        subprocess.run(["git", "add", "src/feature.py"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add feature"], cwd=repo_path, check=True, capture_output=True)

        yield repo_path


class TestFilterRepo:
    """Test git-filter-repo integration."""

    def test_filter_repo_not_a_repo(self):
        """Test error when path is not a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                filter_repo(Path(tmpdir), ["src"])
            assert "not a git repository" in str(exc_info.value).lower()

    @patch("subprocess.run")
    @patch("gitta.filter.run_git")
    def test_filter_repo_tool_not_installed(self, mock_run_git, mock_subprocess_run):
        """Test error when git-filter-repo is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".git").mkdir()

            # Mock successful clone
            mock_run_git.return_value = None

            # Mock filter-repo not found
            mock_subprocess_run.side_effect = FileNotFoundError()

            with pytest.raises(RuntimeError) as exc_info:
                filter_repo(repo_path, ["src"])

            assert "git-filter-repo not found" in str(exc_info.value)
            assert "metta install filter-repo" in str(exc_info.value)

    @pytest.mark.skipif(
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True).returncode != 0,
        reason="git-filter-repo not installed",
    )
    def test_filter_repo_single_directory(self, temp_repo_with_files):
        """Test filtering to a single directory."""
        # This is an integration test that requires git-filter-repo
        result_path = filter_repo(temp_repo_with_files, ["src/"])

        assert result_path.exists()
        assert (result_path / ".git").exists()

        # Check that only src files remain
        files = list(result_path.rglob("*"))
        file_names = [f.relative_to(result_path).as_posix() for f in files if f.is_file()]

        assert any("main.py" in name for name in file_names)
        assert any("utils.py" in name for name in file_names)
        assert any("feature.py" in name for name in file_names)

        # These should be filtered out
        assert not any("README.md" in name for name in file_names)
        assert not any("test_main.py" in name for name in file_names)
        assert not any("LICENSE" in name for name in file_names)

    @pytest.mark.skipif(
        subprocess.run(["git", "filter-repo", "--version"], capture_output=True).returncode != 0,
        reason="git-filter-repo not installed",
    )
    def test_filter_repo_multiple_paths(self, temp_repo_with_files):
        """Test filtering to multiple directories."""
        result_path = filter_repo(temp_repo_with_files, ["src/", "docs/"])

        assert result_path.exists()

        # Check files
        files = list(result_path.rglob("*"))
        file_names = [f.relative_to(result_path).as_posix() for f in files if f.is_file()]

        # Should have src and docs files
        assert any("main.py" in name for name in file_names)
        assert any("README.md" in name for name in file_names)

        # Should not have tests or root files
        assert not any("test_main.py" in name for name in file_names)
        assert not any("LICENSE" in name for name in file_names)

    @patch("subprocess.run")
    @patch("gitta.filter.run_git")
    @patch("gitta.filter.get_file_list")
    def test_filter_repo_empty_result(self, mock_get_file_list, mock_run_git, mock_subprocess_run):
        """Test error when filtering results in empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".git").mkdir()

            # Mock successful operations until empty check
            mock_run_git.return_value = None
            mock_subprocess_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b""
            )

            # Mock empty file list
            mock_get_file_list.return_value = []

            with pytest.raises(RuntimeError) as exc_info:
                filter_repo(repo_path, ["nonexistent/"])

            assert "empty" in str(exc_info.value).lower()

    @patch("subprocess.run")
    @patch("gitta.filter.run_git")
    def test_filter_repo_command_failure(self, mock_run_git, mock_subprocess_run):
        """Test handling of git-filter-repo command failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".git").mkdir()

            # Mock successful clone
            mock_run_git.return_value = None

            # Mock filter-repo version check success, but command failure
            version_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            filter_result = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"Error: invalid path"
            )
            mock_subprocess_run.side_effect = [version_result, filter_result]

            with pytest.raises(RuntimeError) as exc_info:
                filter_repo(repo_path, ["src/"])

            assert "git-filter-repo failed" in str(exc_info.value)
            assert "invalid path" in str(exc_info.value)


class TestFilterRepoWithRealGit:
    """Tests that use real git operations but mock filter-repo."""

    @patch("subprocess.run")
    def test_filter_repo_clone_failure(self, mock_subprocess_run, temp_repo_with_files):
        """Test handling of clone failure."""

        # Let real git handle everything except filter-repo check
        def side_effect(*args, **kwargs):
            if args[0] == ["git", "filter-repo", "--version"]:
                raise FileNotFoundError()
            else:
                # Pass through to real subprocess
                return subprocess.run(*args, **kwargs)

        mock_subprocess_run.side_effect = side_effect

        # Temporarily break the repo to cause clone to fail
        import shutil

        git_dir = temp_repo_with_files / ".git"
        backup_dir = temp_repo_with_files / ".git_backup"
        shutil.move(str(git_dir), str(backup_dir))

        try:
            with pytest.raises(RuntimeError) as exc_info:
                filter_repo(temp_repo_with_files, ["src/"])
            assert "Failed to clone" in str(exc_info.value)
        finally:
            # Restore the repo
            if backup_dir.exists():
                shutil.move(str(backup_dir), str(git_dir))
