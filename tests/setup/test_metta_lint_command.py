import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.setup


def test_lint_with_unsupported_files_exits_cleanly(tmp_path: pathlib.Path) -> None:
    """Test that metta lint with unsupported file extensions exits cleanly without formatting the entire repo."""
    # Create a test file with unsupported extension
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file\n")

    # Run metta lint on the unsupported file
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "lint", str(test_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should exit successfully with a message
    assert result.returncode == 0
    assert "No files with supported extensions found" in result.stdout


def test_lint_staged_with_only_unsupported_files_exits_cleanly(tmp_path: pathlib.Path) -> None:
    """Test that metta lint --staged with only unsupported files exits cleanly."""
    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create and stage an unsupported file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file\n")
    subprocess.run(["git", "add", "test.txt"], cwd=tmp_path, check=True, capture_output=True)

    # Run metta lint --staged
    result = subprocess.run(
        [sys.executable, "-m", "metta.setup.metta_cli", "lint", "--staged"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    # Should exit successfully with a message
    assert result.returncode == 0
    assert "No files with supported extensions found" in result.stdout


def test_lint_check_mode_with_unsupported_formatter() -> None:
    """Test that metta lint --check returns False when formatter doesn't support check mode."""
    import metta.setup.tools.code_formatters

    # Create a formatter without check command
    formatter = metta.setup.tools.code_formatters.FormatterConfig(
        name="test-formatter",
        format_cmd=["echo", "formatting"],
        check_cmd=None,  # No check command
    )

    # Run with check_only=True
    result = metta.setup.tools.code_formatters.run_formatter("test", formatter, pathlib.Path.cwd(), check_only=True)

    # Should return False to indicate check couldn't be performed
    assert result is False


def test_lint_check_mode_with_supported_formatter() -> None:
    """Test that metta lint --check works correctly with formatters that support it."""

    # Create a formatter with check command that succeeds
    formatter = metta.setup.tools.code_formatters.FormatterConfig(
        name="test-formatter",
        format_cmd=["true"],
        check_cmd=["true"],  # Command that always succeeds
    )

    # Run with check_only=True
    result = metta.setup.tools.code_formatters.run_formatter("test", formatter, pathlib.Path.cwd(), check_only=True)

    # Should return True since check command succeeded
    assert result is True
