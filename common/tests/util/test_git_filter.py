import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from metta.common.util.git import get_commit_count, get_file_list
from metta.common.util.git_filter import filter_repo


class TestFilterRepo:
    """Test the filter_repo functionality."""

    @pytest.fixture
    def source_repo(self):
        """Create a source repository with test structure."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "source"
        repo_path.mkdir()

        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create structure matching our filter paths
        (repo_path / "mettagrid").mkdir()
        (repo_path / "mettagrid" / "core.py").write_text("# mettagrid core")
        (repo_path / "mettagrid" / "utils.py").write_text("# mettagrid utils")

        (repo_path / "mettascope").mkdir()
        (repo_path / "mettascope" / "viz.py").write_text("# mettascope viz")

        # Create files that should be filtered out
        (repo_path / "other_module").mkdir()
        (repo_path / "other_module" / "file.py").write_text("# should be filtered")
        (repo_path / "README.md").write_text("# Root readme")

        # Commit everything
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

        # Add more commits to test history preservation
        (repo_path / "mettagrid" / "new_file.py").write_text("# new file")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Add new file to mettagrid"], cwd=repo_path, check=True)

        # Add a tag
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo_path, check=True)

        yield repo_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.skipif(shutil.which("git-filter-repo") is None, reason="git-filter-repo not installed")
    def test_filter_repo_basic(self, source_repo):
        """Test basic filter_repo functionality."""
        # Filter repository
        filtered_path = filter_repo(source_repo, ["mettagrid/", "mettascope/"])

        try:
            # Verify filtered repository
            files = get_file_list(filtered_path)

            # Check that only mettagrid and mettascope files are present
            for file in files:
                assert file.startswith("mettagrid/") or file.startswith("mettascope/"), (
                    f"Unexpected file in filtered repo: {file}"
                )

            # Check specific files
            expected_files = {"mettagrid/core.py", "mettagrid/utils.py", "mettagrid/new_file.py", "mettascope/viz.py"}
            assert set(files) == expected_files

            # Check that history is preserved (2 commits)
            assert get_commit_count(filtered_path) == 2

            # Check that tag is preserved
            result = subprocess.run(["git", "tag", "-l"], cwd=filtered_path, capture_output=True, text=True, check=True)
            tags = result.stdout.strip().split("\n") if result.stdout.strip() else []
            assert "v1.0.0" in tags

        finally:
            # Cleanup
            shutil.rmtree(filtered_path.parent)

    @pytest.mark.skipif(shutil.which("git-filter-repo") is None, reason="git-filter-repo not installed")
    def test_filter_repo_single_path(self, source_repo):
        """Test filter_repo with single path."""
        # Filter repository to single path
        filtered_path = filter_repo(source_repo, ["mettagrid/"])

        try:
            # Verify content
            files = get_file_list(filtered_path)

            # Should only have mettagrid files
            assert all(f.startswith("mettagrid/") for f in files)
            assert len(files) == 3  # core.py, utils.py, new_file.py

        finally:
            # Cleanup
            shutil.rmtree(filtered_path.parent)

    @pytest.mark.skipif(shutil.which("git-filter-repo") is None, reason="git-filter-repo not installed")
    def test_filter_repo_empty_result(self, source_repo):
        """Test filter_repo when result would be empty."""
        # Try to filter non-existent path
        with pytest.raises(RuntimeError, match="Filtered repository is empty"):
            filter_repo(source_repo, ["nonexistent/"])

    def test_filter_repo_missing_tool(self, source_repo, monkeypatch):
        """Test filter_repo when git-filter-repo is not installed."""
        # Mock subprocess.run to simulate missing git-filter-repo
        original_run = subprocess.run

        def mock_run(cmd, *args, **kwargs):
            if len(cmd) >= 3 and cmd[0:3] == ["git", "filter-repo", "--version"]:
                raise subprocess.CalledProcessError(1, cmd, stderr="command not found")
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="git-filter-repo not found"):
            filter_repo(source_repo, ["mettagrid/"])

