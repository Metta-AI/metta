"""Tests for metta.common.util.git module."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from metta.common.util.git import (
    GitError,
    get_branch_commit,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_matched_pr,
    get_remote_url,
    has_unstaged_changes,
    is_commit_pushed,
    is_metta_ai_repo,
    run_gh,
    run_git,
    run_git_in_dir,
    run_git_with_cwd,
    validate_git_ref,
)


class TestGitError:
    """Test cases for GitError exception."""

    def test_git_error_inheritance(self):
        """Test that GitError inherits from Exception."""
        error = GitError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


class TestRunGitCommands:
    """Test cases for git command execution functions."""

    @patch('subprocess.run')
    def test_run_git_with_cwd_success(self, mock_run):
        """Test successful git command execution with cwd."""
        mock_result = Mock()
        mock_result.stdout = "output\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = run_git_with_cwd(["status"], "/some/path")

        assert result == "output"
        mock_run.assert_called_once_with(
            ["git", "status"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/some/path"
        )

    @patch('subprocess.run')
    def test_run_git_with_cwd_called_process_error(self, mock_run):
        """Test git command failure handling."""
        error = subprocess.CalledProcessError(1, "git", stderr="fatal: error")
        mock_run.side_effect = error

        with pytest.raises(GitError, match="Git command failed \\(1\\): fatal: error"):
            run_git_with_cwd(["status"])

    @patch('subprocess.run')
    def test_run_git_with_cwd_file_not_found(self, mock_run):
        """Test git not installed error handling."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(GitError, match="Git is not installed!"):
            run_git_with_cwd(["status"])

    @patch('metta.common.util.git.run_git_with_cwd')
    def test_run_git(self, mock_run_git_with_cwd):
        """Test run_git wrapper function."""
        mock_run_git_with_cwd.return_value = "test output"

        result = run_git("status", "--short")

        assert result == "test output"
        mock_run_git_with_cwd.assert_called_once_with(["status", "--short"], None)

    @patch('metta.common.util.git.run_git_with_cwd')
    def test_run_git_in_dir(self, mock_run_git_with_cwd):
        """Test run_git_in_dir wrapper function."""
        mock_run_git_with_cwd.return_value = "test output"

        result = run_git_in_dir("/some/dir", "log", "--oneline")

        assert result == "test output"
        mock_run_git_with_cwd.assert_called_once_with(["log", "--oneline"], "/some/dir")

    @patch('subprocess.run')
    def test_run_gh_success(self, mock_run):
        """Test successful GitHub CLI command execution."""
        mock_result = Mock()
        mock_result.stdout = "github output\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = run_gh("repo", "list")

        assert result == "github output"

    @patch('subprocess.run')
    def test_run_gh_called_process_error(self, mock_run):
        """Test GitHub CLI command failure handling."""
        error = subprocess.CalledProcessError(2, "gh", stderr="auth required")
        mock_run.side_effect = error

        with pytest.raises(GitError, match="GitHub CLI command failed \\(2\\): auth required"):
            run_gh("repo", "list")

    @patch('subprocess.run')
    def test_run_gh_file_not_found(self, mock_run):
        """Test GitHub CLI not installed error handling."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(GitError, match="GitHub CLI \\(gh\\) is not installed!"):
            run_gh("repo", "list")


class TestGitInfoFunctions:
    """Test cases for git information functions."""

    @patch('metta.common.util.git.run_git')
    def test_get_current_branch_success(self, mock_run_git):
        """Test getting current branch successfully."""
        mock_run_git.return_value = "main"

        result = get_current_branch()

        assert result == "main"
        mock_run_git.assert_called_once_with("symbolic-ref", "--short", "HEAD")

    @patch('metta.common.util.git.run_git')
    def test_get_current_branch_not_git_repo(self, mock_run_git):
        """Test get_current_branch when not in git repository."""
        mock_run_git.side_effect = GitError("not a git repository")

        with pytest.raises(ValueError, match="Not in a git repository"):
            get_current_branch()

    @patch('metta.common.util.git.get_current_commit')
    @patch('metta.common.util.git.run_git')
    def test_get_current_branch_detached_head(self, mock_run_git, mock_get_commit):
        """Test get_current_branch when in detached HEAD state."""
        mock_run_git.side_effect = GitError("HEAD is not a symbolic ref")
        mock_get_commit.return_value = "abc123"

        result = get_current_branch()

        assert result == "abc123"
        mock_get_commit.assert_called_once()

    @patch('metta.common.util.git.run_git')
    def test_get_current_branch_other_error(self, mock_run_git):
        """Test get_current_branch with other GitError."""
        mock_run_git.side_effect = GitError("some other error")

        with pytest.raises(GitError, match="some other error"):
            get_current_branch()

    @patch('metta.common.util.git.run_git')
    def test_get_current_commit(self, mock_run_git):
        """Test getting current commit hash."""
        mock_run_git.return_value = "abc123def456"

        result = get_current_commit()

        assert result == "abc123def456"
        mock_run_git.assert_called_once_with("rev-parse", "HEAD")

    @patch('metta.common.util.git.run_git')
    def test_get_branch_commit_local_branch(self, mock_run_git):
        """Test getting commit for local branch."""
        mock_run_git.return_value = "def456abc789"

        result = get_branch_commit("feature-branch")

        assert result == "def456abc789"
        mock_run_git.assert_called_once_with("rev-parse", "--verify", "feature-branch")

    @patch('metta.common.util.git.run_git')
    def test_get_branch_commit_remote_branch_with_fetch(self, mock_run_git):
        """Test getting commit for remote branch with fetch."""
        def mock_git_calls(*args):
            if args == ("fetch", "--quiet"):
                return ""
            elif args == ("rev-parse", "--verify", "origin/main"):
                return "remote123hash"
            return ""

        mock_run_git.side_effect = mock_git_calls

        result = get_branch_commit("origin/main")

        assert result == "remote123hash"
        assert mock_run_git.call_count == 2

    @patch('metta.common.util.git.run_git')
    def test_get_branch_commit_fetch_failure(self, mock_run_git):
        """Test get_branch_commit when fetch fails (network issues)."""
        def mock_git_calls(*args):
            if args == ("fetch", "--quiet"):
                raise GitError("network error")
            elif args == ("rev-parse", "--verify", "origin/feature"):
                return "feature123hash"
            return ""

        mock_run_git.side_effect = mock_git_calls

        result = get_branch_commit("origin/feature")

        assert result == "feature123hash"

    @patch('metta.common.util.git.run_git')
    def test_get_commit_message(self, mock_run_git):
        """Test getting commit message."""
        mock_run_git.return_value = "Add new feature\n\nThis commit adds a new feature."

        result = get_commit_message("abc123")

        assert result == "Add new feature\n\nThis commit adds a new feature."
        mock_run_git.assert_called_once_with("log", "-1", "--pretty=%B", "abc123")

    @patch('metta.common.util.git.run_git')
    def test_has_unstaged_changes_true(self, mock_run_git):
        """Test has_unstaged_changes when there are changes."""
        mock_run_git.return_value = " M file1.py\n?? file2.txt"

        result = has_unstaged_changes()

        assert result is True
        mock_run_git.assert_called_once_with("status", "--porcelain")

    @patch('metta.common.util.git.run_git')
    def test_has_unstaged_changes_false(self, mock_run_git):
        """Test has_unstaged_changes when there are no changes."""
        mock_run_git.return_value = ""

        result = has_unstaged_changes()

        assert result is False

    @patch('metta.common.util.git.run_git')
    def test_validate_git_ref_valid(self, mock_run_git):
        """Test validate_git_ref with valid reference."""
        mock_run_git.return_value = "abc123def456"

        result = validate_git_ref("HEAD")

        assert result == "abc123def456"
        mock_run_git.assert_called_once_with("rev-parse", "--verify", "HEAD")

    @patch('metta.common.util.git.run_git')
    def test_validate_git_ref_invalid(self, mock_run_git):
        """Test validate_git_ref with invalid reference."""
        mock_run_git.side_effect = GitError("invalid ref")

        result = validate_git_ref("invalid-ref")

        assert result is None

    @patch('metta.common.util.git.run_git')
    def test_get_remote_url_success(self, mock_run_git):
        """Test getting remote URL successfully."""
        mock_run_git.return_value = "https://github.com/user/repo.git"

        result = get_remote_url()

        assert result == "https://github.com/user/repo.git"
        mock_run_git.assert_called_once_with("remote", "get-url", "origin")

    @patch('metta.common.util.git.run_git')
    def test_get_remote_url_failure(self, mock_run_git):
        """Test get_remote_url when git command fails."""
        mock_run_git.side_effect = GitError("no remote")

        result = get_remote_url()

        assert result is None

    @patch('metta.common.util.git.get_remote_url')
    def test_is_metta_ai_repo_true(self, mock_get_remote_url):
        """Test is_metta_ai_repo when URL matches metta-ai/metta."""
        mock_get_remote_url.return_value = "https://github.com/metta-ai/metta.git"

        result = is_metta_ai_repo()

        assert result is True

    @patch('metta.common.util.git.get_remote_url')
    def test_is_metta_ai_repo_false_different_repo(self, mock_get_remote_url):
        """Test is_metta_ai_repo with different repository."""
        mock_get_remote_url.return_value = "https://github.com/other/repo.git"

        result = is_metta_ai_repo()

        assert result is False

    @patch('metta.common.util.git.get_remote_url')
    def test_is_metta_ai_repo_false_no_remote(self, mock_get_remote_url):
        """Test is_metta_ai_repo when no remote URL."""
        mock_get_remote_url.return_value = None

        result = is_metta_ai_repo()

        assert result is False


class TestIsCommitPushed:
    """Test cases for is_commit_pushed function."""

    @patch('metta.common.util.git.run_git')
    def test_is_commit_pushed_invalid_commit(self, mock_run_git):
        """Test is_commit_pushed with invalid commit hash."""
        mock_run_git.side_effect = GitError("invalid object")

        with pytest.raises(GitError, match="Invalid commit hash: invalid123"):
            is_commit_pushed("invalid123")

    @patch('metta.common.util.git.get_current_branch')
    @patch('metta.common.util.git.run_git')
    def test_is_commit_pushed_fast_path_true(self, mock_run_git, mock_get_branch):
        """Test is_commit_pushed fast path when commit is pushed."""
        def mock_git_calls(*args):
            if args == ("rev-parse", "--verify", "abc123"):
                return "abc123"
            elif args == ("rev-parse", "--abbrev-ref", "main@{u}"):
                return "origin/main"
            elif args == ("merge-base", "--is-ancestor", "abc123", "origin/main"):
                return ""
            return ""

        mock_run_git.side_effect = mock_git_calls
        mock_get_branch.return_value = "main"

        result = is_commit_pushed("abc123")

        assert result is True

    @patch('metta.common.util.git.get_current_branch')
    @patch('metta.common.util.git.run_git')
    def test_is_commit_pushed_fast_path_false(self, mock_run_git, mock_get_branch):
        """Test is_commit_pushed fast path when commit is not pushed."""
        def mock_git_calls(*args):
            if args == ("rev-parse", "--verify", "abc123"):
                return "abc123"
            elif args == ("rev-parse", "--abbrev-ref", "main@{u}"):
                return "origin/main"
            elif args == ("merge-base", "--is-ancestor", "abc123", "origin/main"):
                raise GitError("not an ancestor")
            return ""

        mock_run_git.side_effect = mock_git_calls
        mock_get_branch.return_value = "main"

        result = is_commit_pushed("abc123")

        assert result is False

    @patch('metta.common.util.git.get_current_branch')
    @patch('metta.common.util.git.run_git')
    def test_is_commit_pushed_fallback_true(self, mock_run_git, mock_get_branch):
        """Test is_commit_pushed fallback when no upstream and commit is in remote."""
        def mock_git_calls(*args):
            if args == ("rev-parse", "--verify", "abc123"):
                return "abc123"
            elif args == ("rev-parse", "--abbrev-ref", "main@{u}"):
                raise GitError("no upstream")
            elif args == ("branch", "-r", "--contains", "abc123"):
                return "origin/main\norigin/develop"
            return ""

        mock_run_git.side_effect = mock_git_calls
        mock_get_branch.return_value = "main"

        result = is_commit_pushed("abc123")

        assert result is True

    @patch('metta.common.util.git.get_current_branch')
    @patch('metta.common.util.git.run_git')
    def test_is_commit_pushed_fallback_false(self, mock_run_git, mock_get_branch):
        """Test is_commit_pushed fallback when no upstream and commit not in remote."""
        def mock_git_calls(*args):
            if args == ("rev-parse", "--verify", "abc123"):
                return "abc123"
            elif args == ("rev-parse", "--abbrev-ref", "main@{u}"):
                raise GitError("no upstream")
            elif args == ("branch", "-r", "--contains", "abc123"):
                return ""
            return ""

        mock_run_git.side_effect = mock_git_calls
        mock_get_branch.return_value = "main"

        result = is_commit_pushed("abc123")

        assert result is False


class TestGetMatchedPr:
    """Test cases for get_matched_pr function."""

    @patch('httpx.get')
    def test_get_matched_pr_success(self, mock_get):
        """Test get_matched_pr with successful API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"number": 123, "title": "Add new feature"}
        ]
        mock_get.return_value = mock_response

        result = get_matched_pr("abc123")

        assert result == (123, "Add new feature")

    @patch('httpx.get')
    def test_get_matched_pr_no_prs(self, mock_get):
        """Test get_matched_pr when no PRs found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = get_matched_pr("abc123")

        assert result is None

    @patch('httpx.get')
    def test_get_matched_pr_http_status_error_404(self, mock_get):
        """Test get_matched_pr with 404 error."""
        import httpx
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        error = httpx.HTTPStatusError("404", request=Mock(), response=mock_response)
        mock_get.side_effect = error

        result = get_matched_pr("abc123")

        assert result is None

    @patch('httpx.get')
    def test_get_matched_pr_http_status_error_other(self, mock_get):
        """Test get_matched_pr with non-404 HTTP error."""
        import httpx
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        error = httpx.HTTPStatusError("500", request=Mock(), response=mock_response)
        mock_get.side_effect = error

        with pytest.raises(GitError, match="GitHub API error \\(500\\)"):
            get_matched_pr("abc123")

    @patch('httpx.get')
    def test_get_matched_pr_request_error(self, mock_get):
        """Test get_matched_pr with network error."""
        import httpx
        
        error = httpx.RequestError("Network error")
        mock_get.side_effect = error

        with pytest.raises(GitError, match="Network error while querying GitHub"):
            get_matched_pr("abc123")
