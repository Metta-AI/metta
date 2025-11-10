"""Test that run_tool.py correctly propagates exit codes."""

import subprocess


def test_run_tool_returns_exit_code_1_on_exception():
    """Test that tools/run.py exits with code 1 when a tool raises an exception.

    This test reproduces a bug where tools/run.py was calling run_tool.main()
    without wrapping it in sys.exit(), causing the exit code to be discarded.
    """
    # Run a command that will fail with a validation error
    # assembly_lines currently fails with a ValidationError due to map_builder config issues
    result = subprocess.run(
        ["uv", "run", "./tools/run.py", "train", "assembly_lines", "run=test_exit_code"],
        cwd="/Users/jack/src/metta",
        capture_output=True,
        timeout=30,
    )

    # The process should exit with code 1 (error), not 0 (success)
    assert result.returncode == 1, (
        f"Expected exit code 1 for failed tool invocation, got {result.returncode}. "
        f"Stderr: {result.stderr.decode()[:500]}"
    )

    # Verify that the error message was logged
    combined_output = result.stdout.decode() + result.stderr.decode()
    assert "Tool invocation failed" in combined_output, "Expected error message not found in output"


def test_run_tool_returns_exit_code_0_on_success():
    """Test that tools/run.py exits with code 0 when a tool succeeds."""
    # Run a --dry-run which should succeed
    result = subprocess.run(
        ["uv", "run", "./tools/run.py", "train", "arena_basic_easy_shaped", "--dry-run"],
        cwd="/Users/jack/src/metta",
        capture_output=True,
        timeout=30,
    )

    # The process should exit with code 0 (success)
    assert result.returncode == 0, (
        f"Expected exit code 0 for successful tool invocation, got {result.returncode}. "
        f"Stderr: {result.stderr.decode()[:500]}"
    )
