"""Tests for git_filter module - simplified version with only stable tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import subprocess

from metta.common.util.git_filter import filter_repo


class TestFilterRepo:
    """Test cases for filter_repo function - only stable, high-value tests."""

    def test_filter_repo_not_git_repository(self):
        """Test filter_repo with non-git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            # No .git directory

            with pytest.raises(ValueError, match="Not a git repository"):
                filter_repo(source_path, ["test/path"])

    def test_filter_repo_git_command_not_found(self):
        """Test filter_repo when git-filter-repo command is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()

            with patch("metta.common.util.git_filter.run_git"):
                with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                    mock_subprocess.side_effect = FileNotFoundError("Command not found")

                    with pytest.raises(RuntimeError, match="git-filter-repo not found"):
                        filter_repo(source_path, ["test/path"])

    def test_filter_repo_clone_url_format(self):
        """Test that clone uses correct file:// URL format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()

            with patch("metta.common.util.git_filter.run_git") as mock_run_git:
                with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                    mock_subprocess.side_effect = [
                        Mock(returncode=0),  # version check
                        Mock(returncode=0, stderr="")  # filter success
                    ]

                    with patch("metta.common.util.git_filter.get_file_list", return_value=["file1.txt"]):
                        with patch("metta.common.util.git_filter.get_commit_count", return_value=1):
                            filter_repo(source_path, ["test/"])

                            # Verify clone was called with file:// URL
                            clone_call = mock_run_git.call_args_list[0][0]
                            assert clone_call[0] == "clone"
                            assert clone_call[1] == "--no-local"
                            assert clone_call[2].startswith("file://")

    def test_filter_repo_paths_integration(self):
        """Test successful filtering with multiple paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()

            paths = ["mettagrid/", "mettascope/", "common/"]

            with patch("metta.common.util.git_filter.run_git"):
                with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                    mock_subprocess.side_effect = [
                        Mock(returncode=0),  # version check
                        Mock(returncode=0, stderr="")  # filter success
                    ]

                    with patch("metta.common.util.git_filter.get_file_list", return_value=["file1.txt"]):
                        with patch("metta.common.util.git_filter.get_commit_count", return_value=1):
                            with patch("builtins.print"):
                                filter_repo(source_path, paths)

                            # Verify all paths were included in filter command
                            filter_call = mock_subprocess.call_args_list[1][0][0]
                            for path in paths:
                                assert path in filter_call

    def test_filter_repo_empty_paths_list(self):
        """Test filtering with empty paths list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()

            with patch("metta.common.util.git_filter.run_git"):
                with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                    mock_subprocess.side_effect = [
                        Mock(returncode=0),  # version check
                        Mock(returncode=0, stderr="")  # filter success
                    ]

                    with patch("metta.common.util.git_filter.get_file_list", return_value=["file1.txt"]):
                        with patch("metta.common.util.git_filter.get_commit_count", return_value=1):
                            with patch("builtins.print"):
                                result = filter_repo(source_path, [])

                            # Should still work with no path filters
                            assert result is not None

    def test_filter_repo_filter_command_fails(self):
        """Test filter_repo when the filter command fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()

            # Mock successful clone
            with patch("metta.common.util.git_filter.run_git"):
                with patch("metta.common.util.git_filter.tempfile.mkdtemp") as mock_mkdtemp:
                    # Make mkdtemp return a real directory path within our temp_dir
                    temp_target = str(Path(temp_dir) / "temp_filtered")
                    Path(temp_target).mkdir()  # Create the directory that mkdtemp would create
                    mock_mkdtemp.return_value = temp_target

                    with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                        # Mock successful version check but failed filter
                        mock_subprocess.side_effect = [
                            Mock(returncode=0),  # version check success
                            Mock(returncode=1, stderr="Filter failed")  # filter failure
                        ]

                        with pytest.raises(RuntimeError, match="git-filter-repo failed: Filter failed"):
                            filter_repo(source_path, ["test/path"])
