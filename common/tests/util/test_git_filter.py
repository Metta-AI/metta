"""Tests for metta.common.util.git_filter module."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from metta.common.util.git import GitError
from metta.common.util.git_filter import filter_repo


class TestFilterRepo:
    """Test cases for the filter_repo function."""

    def test_filter_repo_validates_git_repository(self):
        """Test that filter_repo validates the source is a git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "not_a_repo"
            source_path.mkdir()
            
            with pytest.raises(ValueError, match="Not a git repository"):
                filter_repo(source_path, ["some/path"])

    @patch("metta.common.util.git_filter.run_git")
    @patch("metta.common.util.git_filter.tempfile.mkdtemp")
    def test_filter_repo_clone_failure(self, mock_mkdtemp, mock_run_git):
        """Test filter_repo when git clone fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            temp_target = str(Path(temp_dir) / "temp_filtered")
            mock_mkdtemp.return_value = temp_target
            
            # Mock clone failure
            mock_run_git.side_effect = GitError("Clone failed")
            
            with pytest.raises(RuntimeError, match="Failed to clone: Clone failed"):
                filter_repo(source_path, ["test/path"])

    @patch("metta.common.util.git_filter.run_git")
    @patch("metta.common.util.git_filter.tempfile.mkdtemp")
    def test_filter_repo_git_filter_repo_not_found(self, mock_mkdtemp, mock_run_git):
        """Test filter_repo when git-filter-repo is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            temp_target = str(Path(temp_dir) / "temp_filtered")
            mock_mkdtemp.return_value = temp_target
            
            # Mock successful clone
            mock_run_git.side_effect = [None]
            
            with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                # Mock git-filter-repo not found
                mock_subprocess.side_effect = FileNotFoundError("Command not found")
                
                with pytest.raises(RuntimeError, match="git-filter-repo not found"):
                    filter_repo(source_path, ["test/path"])

    @patch("metta.common.util.git_filter.run_git")
    @patch("metta.common.util.git_filter.tempfile.mkdtemp")
    def test_filter_repo_filter_command_fails(self, mock_mkdtemp, mock_run_git):
        """Test filter_repo when the filter command fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            temp_target = str(Path(temp_dir) / "temp_filtered")
            mock_mkdtemp.return_value = temp_target
            
            # Mock successful clone
            mock_run_git.side_effect = [None]
            
            with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                # Mock successful version check but failed filter
                mock_subprocess.side_effect = [
                    Mock(returncode=0),  # version check success
                    Mock(returncode=1, stderr="Filter failed")  # filter failure
                ]
                
                with pytest.raises(RuntimeError, match="git-filter-repo failed: Filter failed"):
                    filter_repo(source_path, ["test/path"])

    @patch("metta.common.util.git_filter.run_git")
    @patch("metta.common.util.git_filter.tempfile.mkdtemp")
    def test_filter_repo_empty_result(self, mock_mkdtemp, mock_run_git):
        """Test filter_repo when filtering results in empty repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            temp_target = str(Path(temp_dir) / "temp_filtered")
            mock_mkdtemp.return_value = temp_target
            
            # Mock successful clone
            mock_run_git.side_effect = [None]
            
            with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                mock_subprocess.side_effect = [
                    Mock(returncode=0),  # version check
                    Mock(returncode=0, stderr="")  # filter success
                ]
                
                with patch("metta.common.util.git_filter.get_file_list", return_value=[]):
                    with pytest.raises(RuntimeError, match="Filtered repository is empty!"):
                        filter_repo(source_path, ["nonexistent/path"])

    @patch("metta.common.util.git_filter.run_git")
    @patch("metta.common.util.git_filter.tempfile.mkdtemp")
    @patch("builtins.print")  # Mock print to avoid output during tests
    def test_filter_repo_success(self, mock_print, mock_mkdtemp, mock_run_git):
        """Test successful filtering operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            temp_target = str(Path(temp_dir) / "temp_filtered")
            filtered_path = Path(temp_target) / "filtered"
            mock_mkdtemp.return_value = temp_target
            
            # Mock successful clone
            mock_run_git.side_effect = [None]
            
            with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                mock_subprocess.side_effect = [
                    Mock(returncode=0),  # version check
                    Mock(returncode=0, stderr="")  # filter success
                ]
                
                with patch("metta.common.util.git_filter.get_file_list", return_value=["file1.txt", "file2.py"]):
                    with patch("metta.common.util.git_filter.get_commit_count", return_value=10):
                        result = filter_repo(source_path, ["mettagrid/", "common/"])
                        
                        assert result == filtered_path
                        
                        # Verify correct subprocess calls
                        assert len(mock_subprocess.call_args_list) == 2
                        
                        # Check version call
                        version_call = mock_subprocess.call_args_list[0][0][0]
                        assert version_call == ["git", "filter-repo", "--version"]
                        
                        # Check filter call structure
                        filter_call = mock_subprocess.call_args_list[1][0][0]
                        assert "git" in filter_call
                        assert "filter-repo" in filter_call
                        assert "--force" in filter_call
                        assert "--path" in filter_call

    def test_filter_repo_version_check_fails(self):
        """Test when git-filter-repo version check fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            (source_path / ".git").mkdir()
            
            with patch("metta.common.util.git_filter.run_git"):
                with patch("metta.common.util.git_filter.subprocess.run") as mock_subprocess:
                    # Mock version check failure
                    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "git filter-repo --version")

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
                            assert str(source_path.absolute()) in clone_call[2]

    def test_filter_repo_multiple_paths(self):
        """Test filtering with multiple paths."""
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