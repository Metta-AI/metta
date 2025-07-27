import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import pytest

from metta.common.util.git import (
    GitError,
    GitRepo,
    get_branch_commit,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    has_unstaged_changes,
    is_commit_pushed,
    run_git,
    validate_git_ref,
)

# Keep in mind that CI will only have a shallow copy of the repository!


def test_get_current_branch():
    branch = get_current_branch()
    assert isinstance(branch, str)
    assert len(branch) > 0


def test_get_current_commit():
    commit = get_current_commit()
    assert isinstance(commit, str)
    assert len(commit) == 40  # SHA-1 hash


def test_run_git_error_propagation():
    with pytest.raises(GitError) as e:
        run_git("branch", "--contains", "invalid-invalid-invalid")
    assert "malformed object name" in str(e.value).lower()


def test_get_branch_commit():
    # Test with current branch
    current_branch = get_current_branch()
    branch_commit = get_branch_commit(current_branch)
    assert isinstance(branch_commit, str)
    assert len(branch_commit) == 40
    assert branch_commit == get_current_commit()

    # Test with invalid branch
    with pytest.raises(GitError):
        get_branch_commit("non-existent-branch-name")


def test_get_commit_message():
    # Get message for current commit
    current_commit = get_current_commit()
    message = get_commit_message(current_commit)
    assert isinstance(message, str)
    assert len(message) > 0

    # Test with invalid commit
    with pytest.raises(GitError):
        get_commit_message("invalid-commit-hash")


def test_has_unstaged_changes():
    # First, ensure we have a clean state
    had_changes = has_unstaged_changes()
    if had_changes:
        subprocess.run(["git", "stash", "push", "-m", "test_stash"], check=True)

    try:
        # Test clean state
        assert not has_unstaged_changes()

        # Create a temporary file to test unstaged changes
        test_file = "test_temp_file.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        try:
            # Should detect untracked file
            assert has_unstaged_changes()
        finally:
            os.remove(test_file)

    finally:
        # Restore state if we stashed
        if had_changes:
            subprocess.run(["git", "stash", "pop"], check=False)


def test_is_commit_pushed():
    # Test with current commit
    current_commit = get_current_commit()
    result = is_commit_pushed(current_commit)
    assert isinstance(result, bool)

    # Test with invalid commit - should raise GitError
    with pytest.raises(GitError):
        is_commit_pushed("invalid-commit-hash")


@pytest.mark.parametrize(
    "ref,expected_valid",
    [
        ("HEAD", True),
        ("non-existent-branch", False),
        ("", False),
        ("invalid..ref", False),
    ],
)
def test_validate_git_ref(ref, expected_valid):
    commit_hash = validate_git_ref(ref)

    if expected_valid:
        assert commit_hash is not None
        assert len(commit_hash) == 40  # SHA-1 hashes are 40 chars
    else:
        assert commit_hash is None


def test_validate_git_ref_with_commit():
    # Test with current commit and short version
    current_commit = get_current_commit()

    # Test with full commit hash
    commit_hash = validate_git_ref(current_commit)
    assert commit_hash == current_commit

    # Test with short commit hash
    commit_hash = validate_git_ref(current_commit[:8])
    assert commit_hash == current_commit  # Git should resolve short hash to full hash


def test_remote_operations():
    # Test operations with remote branches if available
    for remote_ref in ["origin/HEAD", "origin/main", "origin/master"]:
        try:
            # Test get_branch_commit with remote
            commit = get_branch_commit(remote_ref)
            assert isinstance(commit, str)
            assert len(commit) == 40

            # Test validate_git_ref with remote
            commit_hash = validate_git_ref(remote_ref)
            assert commit_hash == commit  # Should match the commit we got from get_branch_commit

            return  # Found at least one valid remote ref
        except GitError:
            continue

    pytest.skip("No remote branches available")


def test_validate_git_ref_returns_commit_hash():
    # Test that validate_git_ref returns the commit hash
    commit_hash = validate_git_ref("HEAD")
    assert commit_hash == get_current_commit()

    # Test invalid ref returns None
    commit_hash = validate_git_ref("invalid-ref")
    assert commit_hash is None


def test_detached_head_fallback():
    # Save current state
    original_branch = get_current_branch()

    try:
        # Detach HEAD
        run_git("checkout", "--detach")

        # Should return commit hash when detached
        result = get_current_branch()
        assert result == get_current_commit()
        assert len(result) == 40
    finally:
        # Restore original branch
        run_git("checkout", original_branch)


class TestGitRepo:
    """Test the GitRepo utility class."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test-repo"
        repo_path.mkdir()
        
        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], 
                      cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], 
                      cwd=repo_path, check=True)
        
        # Create test structure
        (repo_path / "file1.txt").write_text("content1")
        (repo_path / "dir1").mkdir()
        (repo_path / "dir1" / "file2.txt").write_text("content2")
        
        # Initial commit
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], 
                      cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_init_valid_repo(self, temp_repo):
        """Test initializing GitRepo with valid repository."""
        repo = GitRepo(temp_repo)
        assert repo.path == temp_repo
    
    def test_init_invalid_repo(self):
        """Test initializing GitRepo with non-repository path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitRepo(temp_dir)
    
    def test_get_commit_hash(self, temp_repo):
        """Test getting commit hash."""
        repo = GitRepo(temp_repo)
        hash_val = repo.get_commit_hash()
        assert len(hash_val) == 40
        assert all(c in '0123456789abcdef' for c in hash_val)
    
    def test_get_file_list(self, temp_repo):
        """Test getting file list."""
        repo = GitRepo(temp_repo)
        files = repo.get_file_list()
        assert set(files) == {"file1.txt", "dir1/file2.txt"}
    
    def test_get_commit_count(self, temp_repo):
        """Test getting commit count."""
        repo = GitRepo(temp_repo)
        assert repo.get_commit_count() == 1
        
        # Add another commit
        (temp_repo / "file3.txt").write_text("content3")
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Second commit"], 
                      cwd=temp_repo, check=True)
        
        assert repo.get_commit_count() == 2
    
    def test_has_path(self, temp_repo):
        """Test path existence checking."""
        repo = GitRepo(temp_repo)
        assert repo.has_path("file1.txt")
        assert repo.has_path("dir1")
        assert not repo.has_path("nonexistent.txt")
    
    def test_add_remote(self, temp_repo):
        """Test adding a remote."""
        repo = GitRepo(temp_repo)
        
        # Add remote
        repo.add_remote("test-remote", "git@github.com:test/repo.git")
        
        # Verify it was added
        result = repo.run_git(["remote", "-v"])
        assert "test-remote" in result.stdout
        assert "git@github.com:test/repo.git" in result.stdout


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
        subprocess.run(["git", "config", "user.email", "test@example.com"], 
                      cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], 
                      cwd=repo_path, check=True)
        
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
        subprocess.run(["git", "commit", "-m", "Initial commit"], 
                      cwd=repo_path, check=True)
        
        # Add more commits to test history preservation
        (repo_path / "mettagrid" / "new_file.py").write_text("# new file")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Add new file to mettagrid"], 
                      cwd=repo_path, check=True)
        
        # Add a tag
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(
        shutil.which("git-filter-repo") is None,
        reason="git-filter-repo not installed"
    )
    def test_filter_repo_basic(self, source_repo):
        """Test basic filter_repo functionality."""
        repo = GitRepo(source_repo)
        
        # Filter repository
        filtered_path = repo.filter_repo(["mettagrid/", "mettascope/"])
        
        try:
            # Verify filtered repository
            filtered_repo = GitRepo(filtered_path)
            files = filtered_repo.get_file_list()
            
            # Check that only mettagrid and mettascope files are present
            for file in files:
                assert file.startswith("mettagrid/") or file.startswith("mettascope/"), \
                    f"Unexpected file in filtered repo: {file}"
            
            # Check specific files
            expected_files = {
                "mettagrid/core.py",
                "mettagrid/utils.py", 
                "mettagrid/new_file.py",
                "mettascope/viz.py"
            }
            assert set(files) == expected_files
            
            # Check that history is preserved (2 commits)
            assert filtered_repo.get_commit_count() == 2
            
            # Check that tag is preserved
            tags = filtered_repo.get_tags()
            assert "v1.0.0" in tags
            
        finally:
            # Cleanup
            shutil.rmtree(filtered_path.parent)
    
    @pytest.mark.skipif(
        shutil.which("git-filter-repo") is None,
        reason="git-filter-repo not installed"
    )
    def test_filter_repo_single_path(self, source_repo):
        """Test filter_repo with single path."""
        repo = GitRepo(source_repo)
        
        # Filter repository to single path
        filtered_path = repo.filter_repo(["mettagrid/"])
        
        try:
            # Verify content
            filtered_repo = GitRepo(filtered_path)
            files = filtered_repo.get_file_list()
            
            # Should only have mettagrid files
            assert all(f.startswith("mettagrid/") for f in files)
            assert len(files) == 3  # core.py, utils.py, new_file.py
            
        finally:
            # Cleanup
            shutil.rmtree(filtered_path.parent)
    
    def test_filter_repo_empty_result(self, source_repo):
        """Test filter_repo when result would be empty."""
        repo = GitRepo(source_repo)
        
        # Try to filter non-existent path
        with pytest.raises(RuntimeError, match="Filtered repository is empty"):
            repo.filter_repo(["nonexistent/"])
    
    def test_filter_repo_missing_tool(self, source_repo, monkeypatch):
        """Test filter_repo when git-filter-repo is not installed."""
        repo = GitRepo(source_repo)
        
        # Mock subprocess.run to simulate missing git-filter-repo
        original_run = subprocess.run
        
        def mock_run(cmd, *args, **kwargs):
            if cmd[0:3] == ["git", "filter-repo", "--version"]:
                result = subprocess.CompletedProcess(cmd, 1, "", "command not found")
                return result
            return original_run(cmd, *args, **kwargs)
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        
        with pytest.raises(RuntimeError, match="git-filter-repo not found"):
            repo.filter_repo(["mettagrid/"])
