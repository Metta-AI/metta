import subprocess

import pytest

from metta.util.git import (
    GitError,
    commit_exists,
    get_current_branch,
    get_current_commit,
    is_commit_contained,
    run_git,
)

# invalid hash (never valid in any repo)
TEST_FAKE_COMMIT_HASH = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


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


def test_commit_exists_for_current_commit():
    commit = get_current_commit()
    assert commit_exists(commit)


def test_commit_does_not_exist_for_fake_commit():
    assert not commit_exists(TEST_FAKE_COMMIT_HASH), f"Commit {TEST_FAKE_COMMIT_HASH} should not exist"


@pytest.mark.parametrize(
    "hash_input, expected",
    [
        "123",  # too short, likely invalid
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",  # invalid hex
    ],
)
def test_commit_exists_various_hashes(hash_input, expected):
    result = commit_exists(hash_input)
    assert result == expected, f"Expected {expected} for hash {hash_input}, got {result}"


def test_run_git_error_propagation():
    with pytest.raises(GitError) as e:
        run_git("branch", "--contains", "invalid-invalid-invalid")
    assert "malformed object name" in str(e.value).lower()


def test_is_commit_contained():
    commit = get_current_commit()
    branch = get_current_branch()
    remote_branch = f"origin/{branch}"

    subprocess.run(["git", "fetch"], check=True)

    # This might return False, but should not error
    _ = is_commit_contained(remote_branch, commit)
