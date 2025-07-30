import subprocess

import pytest

from metta.common.util.git import (
    GitError,
    get_branch_commit,
    get_commit_count,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    get_file_list,
    is_commit_pushed,
    run_git,
    run_git_in_dir,
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


def test_has_unstaged_changes(tmp_path):
    # Create a temporary git repo for clean testing
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # Create initial commit
    (tmp_path / "initial.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Test clean state
    result = run_git_in_dir(tmp_path, "status", "--porcelain")
    assert result == ""

    # Create a new file to test unstaged changes
    (tmp_path / "test_file.txt").write_text("test content")

    # Should detect untracked file
    result = run_git_in_dir(tmp_path, "status", "--porcelain")
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


def test_run_git_with_cwd(tmp_path):
    # Create a temporary git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # Create and commit a file
    (tmp_path / "test.txt").write_text("test")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Test run_git_in_dir
    result = run_git_in_dir(tmp_path, "rev-parse", "HEAD")
    assert isinstance(result, str)
    assert len(result) == 40

    # Test get_file_list with repo_path
    files = get_file_list(tmp_path)
    assert files == ["test.txt"]

    # Test get_commit_count with repo_path
    count = get_commit_count(tmp_path)
    assert count == 1
