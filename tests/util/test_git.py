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
    run_git,
    validate_git_ref,
)

# Keep in mind that CI will only have a shallow copy of the repository!


def test_get_current_branch():
    branch = get_current_branch()
    assert isinstance(branch, str)
    assert len(branch) > 0
    print(f"Current branch: {branch}")


def test_get_current_commit():
    commit = get_current_commit()
    assert isinstance(commit, str)
    assert len(commit) == 40  # SHA-1 hash
    print(f"Current commit: {commit}")


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

    # Should match current commit on current branch
    current_commit = get_current_commit()
    assert branch_commit == current_commit


def test_get_branch_commit_with_remote():
    # Test with a remote branch (if available)
    try:
        # Try to get origin/HEAD or origin/main
        for branch in ["origin/HEAD", "origin/main", "origin/master"]:
            try:
                commit = get_branch_commit(branch)
                assert isinstance(commit, str)
                assert len(commit) == 40
                break
            except GitError:
                continue
    except GitError:
        # Remote branches might not exist in CI
        pytest.skip("No remote branches available")


def test_get_branch_commit_invalid_branch():
    with pytest.raises(GitError):
        get_branch_commit("non-existent-branch-name")


def test_get_commit_message():
    # Get message for current commit
    current_commit = get_current_commit()
    message = get_commit_message(current_commit)

    assert isinstance(message, str)
    assert len(message) > 0
    print(f"Current commit message: {message[:50]}...")


def test_get_commit_message_invalid():
    with pytest.raises(GitError):
        get_commit_message("invalid-commit-hash")


def test_has_unstaged_changes_clean():
    # First, ensure we have a clean state by stashing if needed
    had_changes = False
    try:
        # Save current state
        had_changes = has_unstaged_changes()
        if had_changes:
            subprocess.run(["git", "stash", "push", "-m", "test_stash"], check=True)

        # Test clean state
        assert not has_unstaged_changes()

    finally:
        # Restore state if we stashed
        if had_changes:
            subprocess.run(["git", "stash", "pop"], check=False)


def test_has_unstaged_changes_with_modifications():
    # Create a temporary file to test unstaged changes
    test_file = "test_temp_file.txt"

    try:
        # Create and add a file
        with open(test_file, "w") as f:
            f.write("test content")

        # Should detect untracked file
        assert has_unstaged_changes()

        # Stage the file
        subprocess.run(["git", "add", test_file], check=True)

        # Modify the file
        with open(test_file, "a") as f:
            f.write("\nmore content")

        # Should still detect unstaged changes
        assert has_unstaged_changes()

    finally:
        # Clean up
        subprocess.run(["git", "reset", "HEAD", test_file], check=False)
        if os.path.exists(test_file):
            os.remove(test_file)


def test_is_commit_pushed():
    # Test with current commit
    current_commit = get_current_commit()

    # This might be True or False depending on CI state
    result = is_commit_pushed(current_commit)
    assert isinstance(result, bool)

    # Test with invalid commit
    assert not is_commit_pushed("invalid-commit-hash")


def test_is_commit_pushed_with_known_old_commit():
    # Try to find an old commit that's likely pushed
    try:
        # Get commits from current branch going back
        commits = run_git("log", "--format=%H", "-n", "10").splitlines()

        if len(commits) > 5:
            # An older commit is more likely to be pushed
            old_commit = commits[-1]
            result = is_commit_pushed(old_commit)
            assert isinstance(result, bool)
    except GitError:
        # Might fail in shallow clones
        pytest.skip("Cannot access commit history")


@pytest.mark.parametrize(
    "test_branch,should_exist",
    [
        ("HEAD", True),
        ("non-existent-branch", False),
    ],
)
def test_get_branch_commit_parametrized(test_branch, should_exist):
    if should_exist:
        commit = get_branch_commit(test_branch)
        assert isinstance(commit, str)
        assert len(commit) == 40
    else:
        with pytest.raises(GitError):
            get_branch_commit(test_branch)


def test_validate_git_ref_with_current_commit():
    current_commit = get_current_commit()
    assert validate_git_ref(current_commit) is True


def test_validate_git_ref_with_current_branch():
    current_branch = get_current_branch()
    assert validate_git_ref(current_branch) is True


def test_validate_git_ref_with_head():
    assert validate_git_ref("HEAD") is True


def test_validate_git_ref_with_head_relative():
    # Test HEAD~1 (if history exists)
    try:
        # First check if HEAD~1 exists
        run_git("rev-parse", "HEAD~1")
        assert validate_git_ref("HEAD~1") is True
    except GitError:
        # Might fail in shallow clones
        pytest.skip("Cannot access commit history")


def test_validate_git_ref_with_remote_branch():
    # Test with remote branches (if available)
    for remote_ref in ["origin/HEAD", "origin/main", "origin/master"]:
        try:
            # Check if this remote ref exists
            run_git("rev-parse", remote_ref)
            assert validate_git_ref(remote_ref) is True
            return  # Found at least one valid remote ref
        except GitError:
            continue

    # If we get here, no remote refs were found
    pytest.skip("No remote branches available")


def test_validate_git_ref_with_invalid_ref():
    invalid_refs = [
        "non-existent-branch",
        "invalid-commit-hash",
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        "refs/heads/does-not-exist",
    ]

    for ref in invalid_refs:
        assert validate_git_ref(ref) is False


def test_validate_git_ref_with_short_commit():
    # Get current commit and test with short version
    current_commit = get_current_commit()
    short_commit = current_commit[:8]

    assert validate_git_ref(short_commit) is True


@pytest.mark.parametrize(
    "ref,should_be_valid",
    [
        ("HEAD", True),
        ("HEAD^", True),
        ("main", None),  # Might or might not exist
        ("master", None),  # Might or might not exist
        ("origin/main", None),  # Might or might not exist
        ("totally-invalid-ref-name", False),
        ("", False),
    ],
)
def test_validate_git_ref_parametrized(ref, should_be_valid):
    is_valid = validate_git_ref(ref)

    if should_be_valid is True:
        assert is_valid is True
    elif should_be_valid is False:
        assert is_valid is False
    else:
        # should_be_valid is None - we don't know if it should exist
        assert isinstance(is_valid, bool)
