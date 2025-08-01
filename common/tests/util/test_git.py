import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from metta.common.util.git import (
    GitError,
    add_remote,
    get_branch_commit,
    get_commit_count,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_file_list,
    get_git_hash_for_remote_task,
    get_latest_commit,
    get_matched_pr,
    get_remote_url,
    is_commit_pushed,
    is_metta_ai_repo,
    run_gh,
    run_git,
    run_git_in_dir,
    validate_git_ref,
)

# Keep in mind that CI will only have a shallow copy of the repository!


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    return tmp_path


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


@pytest.mark.slow
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


def test_has_unstaged_changes(git_repo):
    # Create initial commit
    (git_repo / "initial.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=git_repo, check=True)

    # Test clean state
    result = run_git_in_dir(git_repo, "status", "--porcelain")
    assert result == ""

    # Create a new file to test unstaged changes
    (git_repo / "test_file.txt").write_text("test content")

    # Should detect untracked file
    result = run_git_in_dir(git_repo, "status", "--porcelain")
    assert "test_file.txt" in result


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


def test_get_file_list():
    # Test in current repo
    files = get_file_list()
    assert isinstance(files, list)
    assert len(files) > 0
    assert all(isinstance(f, str) for f in files)


def test_get_commit_count():
    # Test in current repo
    count = get_commit_count()
    assert isinstance(count, int)
    assert count > 0


def test_run_git_with_cwd(git_repo):
    # Create and commit a file
    (git_repo / "test.txt").write_text("test")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=git_repo, check=True)

    # Test run_git_in_dir
    result = run_git_in_dir(git_repo, "rev-parse", "HEAD")
    assert isinstance(result, str)
    assert len(result) == 40

    # Test get_file_list with repo_path
    files = get_file_list(git_repo)
    assert files == ["test.txt"]

    # Test get_commit_count with repo_path
    count = get_commit_count(git_repo)
    assert count == 1


# New tests for previously untested functions


def test_run_gh():
    # Test basic gh command (if gh is installed)
    try:
        result = run_gh("--version")
        assert isinstance(result, str)
        assert "gh version" in result
    except GitError as e:
        if "GitHub CLI (gh) is not installed" in str(e):
            pytest.skip("GitHub CLI not installed")
        else:
            raise


def test_run_gh_error():
    # Test invalid gh command
    with pytest.raises(GitError) as e:
        run_gh("invalid-command-that-does-not-exist")
    assert "GitHub CLI command failed" in str(e.value) or "GitHub CLI (gh) is not installed" in str(e.value)


def test_get_remote_url():
    # Test in current repo
    url = get_remote_url()
    if url:
        assert isinstance(url, str)
        assert url.startswith(("https://", "git@"))


def test_get_remote_url_no_remote(git_repo):
    # Test in repo without remote
    # Need to change to the git_repo directory to test
    original_dir = Path.cwd()
    try:
        import os

        os.chdir(git_repo)
        result = get_remote_url()
        assert result is None
    finally:
        os.chdir(original_dir)


def test_is_metta_ai_repo(monkeypatch):
    # Import the constants to get the exact repo name
    from metta.common.util.git import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO

    # Test with metta-ai repo URLs (using exact format from constants)
    monkeypatch.setattr(
        "metta.common.util.git.get_remote_url",
        lambda: f"https://github.com/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}.git",
    )
    assert is_metta_ai_repo() is True

    monkeypatch.setattr(
        "metta.common.util.git.get_remote_url",
        lambda: f"git@github.com:{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}.git",
    )
    assert is_metta_ai_repo() is True

    # Test with different repo
    monkeypatch.setattr("metta.common.util.git.get_remote_url", lambda: "https://github.com/other/repo.git")
    assert is_metta_ai_repo() is False

    # Test with no remote
    monkeypatch.setattr("metta.common.util.git.get_remote_url", lambda: None)
    assert is_metta_ai_repo() is False


@pytest.mark.network
def test_get_matched_pr(monkeypatch):
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = [{"number": 123, "title": "Test PR"}]
    mock_response.raise_for_status = Mock()

    with patch("httpx.get", return_value=mock_response):
        result = get_matched_pr("fake-commit-hash")
        assert result == (123, "Test PR")

    # Mock empty response (no PRs)
    mock_response.json.return_value = []
    with patch("httpx.get", return_value=mock_response):
        result = get_matched_pr("fake-commit-hash")
        assert result is None

    # Mock 404 response
    mock_404 = Mock()
    mock_404.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found", request=Mock(), response=Mock(status_code=404)
    )
    with patch("httpx.get", return_value=mock_404):
        result = get_matched_pr("fake-commit-hash")
        assert result is None

    # Mock network error
    with patch("httpx.get", side_effect=httpx.RequestError("Network error")):
        with pytest.raises(GitError) as e:
            get_matched_pr("fake-commit-hash")
        assert "Network error" in str(e.value)


def test_get_git_hash_for_remote_task(monkeypatch, caplog):
    logger = logging.getLogger()

    # Test when not in git repo
    monkeypatch.setattr(
        "metta.common.util.git.get_current_commit", Mock(side_effect=ValueError("Not in a git repository"))
    )
    result = get_git_hash_for_remote_task(logger=logger)
    assert result is None
    assert "Not in a git repository" in caplog.text

    # Test when not metta-ai repo
    monkeypatch.setattr("metta.common.util.git.get_current_commit", lambda: "abc123")
    monkeypatch.setattr("metta.common.util.git.is_metta_ai_repo", lambda: False)
    caplog.clear()
    result = get_git_hash_for_remote_task(logger=logger)
    assert result is None
    assert "Origin not set to metta-ai/metta" in caplog.text

    # Test with uncommitted changes (should raise)
    monkeypatch.setattr("metta.common.util.git.is_metta_ai_repo", lambda: True)
    monkeypatch.setattr("metta.common.util.git.has_unstaged_changes", lambda: True)
    with pytest.raises(GitError) as e:
        get_git_hash_for_remote_task()
    assert "uncommitted changes" in str(e.value)

    # Test with uncommitted changes but skip_git_check=True
    caplog.clear()
    # Need to mock is_commit_pushed to avoid git command with fake commit hash
    monkeypatch.setattr("metta.common.util.git.is_commit_pushed", lambda x: True)
    result = get_git_hash_for_remote_task(skip_git_check=True, logger=logger)
    assert result == "abc123"
    assert "Proceeding with uncommitted changes" in caplog.text

    # Test with unpushed commit
    monkeypatch.setattr("metta.common.util.git.has_unstaged_changes", lambda: False)
    monkeypatch.setattr("metta.common.util.git.is_commit_pushed", lambda x: False)
    with pytest.raises(GitError) as e:
        get_git_hash_for_remote_task()
    assert "hasn't been pushed" in str(e.value)

    # Test with unpushed commit but skip_git_check=True
    caplog.clear()
    result = get_git_hash_for_remote_task(skip_git_check=True, skip_cmd="--skip-check", logger=logger)
    assert result == "abc123"
    assert "Proceeding with unpushed commit" in caplog.text

    # Test success case
    monkeypatch.setattr("metta.common.util.git.is_commit_pushed", lambda x: True)
    result = get_git_hash_for_remote_task()
    assert result == "abc123"


@pytest.mark.asyncio
@pytest.mark.network
async def test_get_latest_commit():
    # Mock successful API response
    mock_response = {"sha": "1234567890abcdef1234567890abcdef12345678"}

    class MockResponse:
        def json(self):
            return mock_response

        def raise_for_status(self):
            pass

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return MockResponse()

    with patch("httpx.AsyncClient", MockClient):
        result = await get_latest_commit()
        assert result == "1234567890abcdef1234567890abcdef12345678"

        # Test with different branch
        result = await get_latest_commit("develop")
        assert result == "1234567890abcdef1234567890abcdef12345678"


def test_add_remote(git_repo):
    # Add remote
    add_remote("test-remote", "https://example.com/repo.git", git_repo)

    # Verify remote was added
    result = run_git_in_dir(git_repo, "remote", "get-url", "test-remote")
    assert result == "https://example.com/repo.git"

    # Test replacing existing remote
    add_remote("test-remote", "https://new-url.com/repo.git", git_repo)
    result = run_git_in_dir(git_repo, "remote", "get-url", "test-remote")
    assert result == "https://new-url.com/repo.git"


def test_add_remote_current_directory():
    # Test without repo_path (current directory)
    # Save current remotes
    try:
        existing_remotes = run_git("remote", "-v")
    except GitError:
        existing_remotes = ""

    if "test-temp-remote" not in existing_remotes:
        # Safe to test in current directory
        add_remote("test-temp-remote", "https://temp.example.com/repo.git")
        try:
            result = run_git("remote", "get-url", "test-temp-remote")
            assert result == "https://temp.example.com/repo.git"
        finally:
            # Clean up
            try:
                run_git("remote", "remove", "test-temp-remote")
            except GitError:
                pass


def test_git_not_installed(monkeypatch):
    # Mock subprocess.run to raise FileNotFoundError
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr("subprocess.run", mock_run)

    with pytest.raises(GitError) as e:
        run_git("status")
    assert "Git is not installed" in str(e.value)


def test_empty_repo_edge_cases(git_repo):
    # Test get_file_list on empty repo
    files = get_file_list(git_repo)
    assert files == []

    # Test get_commit_count on empty repo
    count = get_commit_count(git_repo)
    assert count == 0

    # Test get_current_commit on empty repo (should fail)
    with pytest.raises(GitError):
        run_git_in_dir(git_repo, "rev-parse", "HEAD")


def test_git_error_with_different_exit_codes(monkeypatch):
    # Test GitError includes exit code
    def mock_run(*args, **kwargs):
        error = subprocess.CalledProcessError(128, ["git"], stderr="fatal: error")
        raise error

    monkeypatch.setattr("subprocess.run", mock_run)

    with pytest.raises(GitError) as e:
        run_git("status")
    assert "128" in str(e.value)
    assert "fatal: error" in str(e.value)


def test_is_commit_pushed_fast_path(monkeypatch):
    """Test the fast path for is_commit_pushed when upstream is configured."""

    # Mock successful upstream check
    def mock_run_git(*args):
        if args == ("rev-parse", "--abbrev-ref", "main@{u}"):
            return "origin/main"
        elif args == ("merge-base", "--is-ancestor", "test-commit", "origin/main"):
            return ""  # Success (exit code 0)
        else:
            raise GitError("Unexpected git command")

    monkeypatch.setattr("metta.common.util.git.get_current_branch", lambda: "main")
    monkeypatch.setattr("metta.common.util.git.run_git", mock_run_git)

    assert is_commit_pushed("test-commit") is True

    # Test when commit is not an ancestor
    def mock_run_git_not_ancestor(*args):
        if args == ("rev-parse", "--abbrev-ref", "main@{u}"):
            return "origin/main"
        elif args == ("merge-base", "--is-ancestor", "test-commit", "origin/main"):
            raise GitError("Not an ancestor")
        else:
            raise GitError("Unexpected git command")

    monkeypatch.setattr("metta.common.util.git.run_git", mock_run_git_not_ancestor)
    assert is_commit_pushed("test-commit") is False


def test_get_branch_commit_with_remote_fetch(monkeypatch):
    """Test that get_branch_commit fetches for remote branches."""
    fetch_called = False

    def mock_run_git(*args):
        nonlocal fetch_called
        if args == ("fetch", "--quiet"):
            fetch_called = True
            return ""
        elif args == ("rev-parse", "--verify", "origin/main"):
            return "abcd1234" * 5  # 40 char hash
        else:
            raise GitError(f"Unexpected git command: {args}")

    monkeypatch.setattr("metta.common.util.git.run_git", mock_run_git)

    result = get_branch_commit("origin/main")
    assert result == "abcd1234" * 5
    assert fetch_called

    # Test that fetch failure is non-fatal
    fetch_called = False

    def mock_run_git_fetch_fails(*args):
        nonlocal fetch_called
        if args == ("fetch", "--quiet"):
            fetch_called = True
            raise GitError("Network error")
        elif args == ("rev-parse", "--verify", "origin/main"):
            return "abcd1234" * 5
        else:
            raise GitError(f"Unexpected git command: {args}")

    monkeypatch.setattr("metta.common.util.git.run_git", mock_run_git_fetch_fails)

    result = get_branch_commit("origin/main")
    assert result == "abcd1234" * 5
    assert fetch_called
