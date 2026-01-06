"""Test that run_tool.py correctly propagates exit codes."""

import subprocess

from metta.common.util.fs import get_repo_root


def test_run_tool_returns_exit_code_1_on_exception():
    """Test that tools/run.py exits with code 1 when a tool raises an exception.

    This test verifies that run_tool.main() is wrapped in sys.exit() so that
    exceptions properly result in non-zero exit codes.
    """
    # Run a command with a non-existent recipe to trigger an error

    result = subprocess.run(
        ["uv", "run", "./tools/run.py", "train", "nonexistent_recipe_that_does_not_exist"],
        cwd=get_repo_root(),
        capture_output=True,
        timeout=120,
    )

    # The process should exit with a non-zero code (error), not 0 (success)
    # Exit code 1 = tool invocation failed, 2 = usage error
    assert result.returncode != 0, (
        f"Expected non-zero exit code for failed tool invocation, got {result.returncode}. "
        f"Stderr: {result.stderr.decode()[:500]}"
    )

    # Verify that an error message was logged
    combined_output = result.stdout.decode() + result.stderr.decode()
    has_error = (
        "nonexistent_recipe" in combined_output.lower()
        or "not found" in combined_output.lower()
        or "error" in combined_output.lower()
    )
    assert has_error, "Expected error message not found in output"


def test_run_tool_returns_exit_code_0_on_success():
    """Test that tools/run.py exits with code 0 when a tool succeeds."""

    # Run a --dry-run which should succeed
    result = subprocess.run(
        ["uv", "run", "./tools/run.py", "train", "arena_basic_easy_shaped", "--dry-run"],
        cwd=get_repo_root(),
        capture_output=True,
        timeout=120,
    )

    # The process should exit with code 0 (success)
    assert result.returncode == 0, (
        f"Expected exit code 0 for successful tool invocation, got {result.returncode}. "
        f"Stderr: {result.stderr.decode()[:500]}"
    )
