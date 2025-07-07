import pytest

from app_backend.git_client import GitClient, GitCommit, GitError, run_git


class TestGitUtilities:
    """Tests for git utility functions."""

    def test_run_git_success(self):
        """Test successful git command execution."""
        result = run_git("rev-parse", "HEAD")
        assert isinstance(result, str)
        assert len(result) == 40  # SHA-1 hash

    def test_run_git_failure(self):
        """Test git command failure handling."""
        with pytest.raises(GitError) as exc_info:
            run_git("invalid-command", "invalid-args")
        assert "Git command failed" in str(exc_info.value)


class TestGitClient:
    """Tests for GitClient."""

    def test_commit_exists_valid(self):
        """Test commit existence check with valid commit."""
        git_client = GitClient()
        current_commit = run_git("rev-parse", "HEAD")
        assert git_client.commit_exists(current_commit)

    def test_commit_exists_invalid(self):
        """Test commit existence check with invalid commit."""
        git_client = GitClient()
        assert not git_client.commit_exists("invalid-commit-hash")

    def test_get_current_commit(self):
        """Test getting current commit."""
        git_client = GitClient()
        commit = git_client.get_current_commit()
        assert isinstance(commit, str)
        assert len(commit) == 40

    def test_get_current_branch(self):
        """Test getting current branch."""
        git_client = GitClient()
        branch = git_client.get_current_branch()
        assert isinstance(branch, str)
        assert len(branch) > 0

    def test_get_merge_base_with_main(self):
        """Test getting merge base with main branch."""
        git_client = GitClient()
        current_commit = run_git("rev-parse", "HEAD")
        merge_base = git_client.get_merge_base(current_commit, "main")
        assert isinstance(merge_base, str)
        assert len(merge_base) == 40  # SHA-1 hash

    def test_get_commit_range_single_commit(self):
        """Test getting commit range for single commit."""
        git_client = GitClient()
        current_commit = run_git("rev-parse", "HEAD")

        commits = git_client.get_commit_range(current_commit)
        assert isinstance(commits, list)
        if commits:  # Only check if we have commits
            assert all(isinstance(c, GitCommit) for c in commits)
            assert all(len(c.hash) == 40 for c in commits)

    def test_get_commit_range_invalid_commit(self):
        """Test getting commit range for invalid commit."""
        git_client = GitClient()
        with pytest.raises(ValueError) as exc_info:
            git_client.get_commit_range("invalid-commit-hash")
        assert "Could not retrieve commit history" in str(exc_info.value)

    def test_get_commit_range_known_commit(self):
        """Test getting commit range for known commit hash."""
        git_client = GitClient()

        # Use a known commit hash from the current repository
        # This hash should exist in the metta project history
        known_commit = "2308eaf792dc19726ba7056cda0a32f5b3cacf3a"
        # Use a known base commit that should be in main (commit before the target)
        known_base = "3af59ac4c5493815c141cd4db7435e8b57f67457"

        # Only run test if both commits exist locally
        if git_client.commit_exists(known_commit) and git_client.commit_exists(known_base):
            commits = git_client.get_commit_range(known_commit, known_base)
            assert isinstance(commits, list)
            assert len(commits) > 0

            # Check that the commit we're looking for is in the range
            commit_hashes = [c.hash for c in commits]
            assert known_commit in commit_hashes

            # Verify the expected commit message exists
            expected_commit = next(c for c in commits if c.hash == known_commit)
            assert expected_commit.message == "Add training run description"
            assert expected_commit.author == "Paul Tsier"

    def test_get_commit_range_edge_case_same_commit(self):
        """Test get_commit_range when commit_hash equals base_branch."""
        git_client = GitClient()

        # Use a known commit hash from the current repository
        known_commit = "2308eaf792dc19726ba7056cda0a32f5b3cacf3a"

        # Only run test if the commit exists locally
        if git_client.commit_exists(known_commit):
            # Test edge case: commit_hash equals base_branch
            commits = git_client.get_commit_range(known_commit, known_commit)
            assert isinstance(commits, list)
            assert len(commits) == 1  # Should return the single commit
            assert commits[0].hash == known_commit
            assert commits[0].message == "Add training run description"
            assert commits[0].author == "Paul Tsier"
        else:
            pytest.skip("Known commits not available locally")


class TestGitCommit:
    """Tests for GitCommit class."""

    def test_git_commit_creation(self):
        """Test GitCommit object creation."""
        commit = GitCommit(
            hash="abcd1234567890abcd1234567890abcd12345678",
            message="Test commit message",
            author="Test Author",
            date="2024-01-01",
        )

        assert commit.hash == "abcd1234567890abcd1234567890abcd12345678"
        assert commit.message == "Test commit message"
        assert commit.author == "Test Author"
        assert commit.date == "2024-01-01"

    def test_git_commit_repr(self):
        """Test GitCommit string representation."""
        commit = GitCommit(
            hash="abcd1234567890abcd1234567890abcd12345678",
            message="This is a very long commit message that should be truncated in the repr",
            author="Test Author",
            date="2024-01-01",
        )

        repr_str = repr(commit)
        assert "abcd1234" in repr_str  # Short hash
        assert "This is a very long commit message that should be ..." in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
