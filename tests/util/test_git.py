import os
import subprocess

import pytest

from metta.util.git import (
    GitError,
    get_branch_commit,
    get_commit_message,
    get_current_branch,
    get_current_commit,
    has_unstaged_changes,
    is_commit_pushed,
    ref_to_hash,
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

    # Test with invalid commit
    assert not is_commit_pushed("invalid-commit-hash")


@pytest.mark.parametrize(
    "ref,expected_valid",
    [
        ("HEAD", True),
        ("non-existent-branch", False),
        ("", False),
    ],
)
def test_validate_git_ref(ref, expected_valid):
    assert validate_git_ref(ref) is expected_valid


def test_validate_git_ref_with_commit():
    # Test with current commit and short version
    current_commit = get_current_commit()
    assert validate_git_ref(current_commit) is True
    assert validate_git_ref(current_commit[:8]) is True


def test_ref_to_hash():
    current_commit = get_current_commit()

    # Test with valid refs
    assert ref_to_hash("HEAD") == current_commit
    assert ref_to_hash(current_commit) == current_commit
    assert ref_to_hash(get_current_branch()) == current_commit

    # Test with short commit
    short_commit = current_commit[:8]
    assert ref_to_hash(short_commit) == current_commit

    # Test with invalid refs
    assert ref_to_hash("non-existent-branch") is None
    assert ref_to_hash("") is None
    assert ref_to_hash("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz") is None


def test_ref_to_hash_with_history():
    # Test HEAD~1 if history exists
    try:
        run_git("rev-parse", "HEAD~1")
        result = ref_to_hash("HEAD~1")
        assert result is not None
        assert len(result) == 40
        assert result != get_current_commit()
    except GitError:
        pytest.skip("Cannot access commit history")


def test_remote_operations():
    # Test operations with remote branches if available
    for remote_ref in ["origin/HEAD", "origin/main", "origin/master"]:
        try:
            # Test get_branch_commit with remote
            commit = get_branch_commit(remote_ref)
            assert isinstance(commit, str)
            assert len(commit) == 40

            # Test validate_git_ref with remote
            assert validate_git_ref(remote_ref) is True

            # Test ref_to_hash with remote
            assert ref_to_hash(remote_ref) == commit

            return  # Found at least one valid remote ref
        except GitError:
            continue

    pytest.skip("No remote branches available")
