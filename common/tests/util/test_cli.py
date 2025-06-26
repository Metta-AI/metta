"""Tests for metta.common.util.cli module."""

import subprocess
import sys
from unittest.mock import patch

import pytest

from metta.common.util.cli import die, get_user_confirmation, sh


class TestGetUserConfirmation:
    """Test cases for the get_user_confirmation function."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_confirmation_yes_responses(self, mock_print, mock_input):
        """Test that various 'yes' responses return True."""
        yes_responses = ["", "y", "Y", "yes", "YES", "Yes"]

        for response in yes_responses:
            mock_input.return_value = response
            result = get_user_confirmation("Test prompt")
            assert result is True
            mock_input.assert_called_with("Test prompt (Y/n): ")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_confirmation_no_responses(self, mock_print, mock_input):
        """Test that various 'no' responses return False."""
        no_responses = ["n", "N", "no", "NO", "No", "nope", "anything_else"]

        for response in no_responses:
            mock_input.return_value = response
            result = get_user_confirmation("Test prompt")
            assert result is False
            mock_input.assert_called_with("Test prompt (Y/n): ")
            mock_print.assert_called()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_confirmation_with_whitespace(self, mock_print, mock_input):
        """Test that responses with whitespace are handled correctly."""
        # Test whitespace around valid responses
        mock_input.return_value = "  y  "
        result = get_user_confirmation()
        assert result is True

        mock_input.return_value = "  n  "
        result = get_user_confirmation()
        assert result is False

    @patch("builtins.input")
    def test_confirmation_default_prompt(self, mock_input):
        """Test that the default prompt is used when none is provided."""
        mock_input.return_value = "y"
        get_user_confirmation()
        mock_input.assert_called_with("Should we proceed? (Y/n): ")

    @patch("builtins.input")
    def test_confirmation_custom_prompt(self, mock_input):
        """Test that custom prompts are used correctly."""
        custom_prompt = "Do you want to continue with the operation?"
        mock_input.return_value = "y"
        get_user_confirmation(custom_prompt)
        mock_input.assert_called_with(f"{custom_prompt} (Y/n): ")


class TestSh:
    """Test cases for the sh function."""

    @patch("subprocess.check_output")
    def test_sh_successful_command(self, mock_check_output):
        """Test sh function with a successful command."""
        mock_check_output.return_value = "command output\n"

        result = sh(["echo", "hello"])

        assert result == "command output"
        mock_check_output.assert_called_once_with(["echo", "hello"], text=True)

    @patch("subprocess.check_output")
    def test_sh_with_kwargs(self, mock_check_output):
        """Test sh function with additional keyword arguments."""
        mock_check_output.return_value = "output\n"

        result = sh(["ls"], cwd="/tmp", env={"PATH": "/usr/bin"})

        assert result == "output"
        mock_check_output.assert_called_once_with(["ls"], text=True, cwd="/tmp", env={"PATH": "/usr/bin"})

    @patch("subprocess.check_output")
    def test_sh_strips_whitespace(self, mock_check_output):
        """Test that sh function strips leading and trailing whitespace."""
        mock_check_output.return_value = "  \n  output with spaces  \n  "

        result = sh(["command"])

        assert result == "output with spaces"

    @patch("subprocess.check_output")
    def test_sh_command_failure(self, mock_check_output):
        """Test sh function when command fails."""
        mock_check_output.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["false"], output="error output"
        )

        with pytest.raises(subprocess.CalledProcessError):
            sh(["false"])

    @patch("subprocess.check_output")
    def test_sh_empty_output(self, mock_check_output):
        """Test sh function with empty command output."""
        mock_check_output.return_value = ""

        result = sh(["true"])

        assert result == ""

    @patch("subprocess.check_output")
    def test_sh_multiline_output(self, mock_check_output):
        """Test sh function with multiline output."""
        mock_check_output.return_value = "line1\nline2\nline3\n"

        result = sh(["command"])

        assert result == "line1\nline2\nline3"


class TestDie:
    """Test cases for the die function."""

    @patch("sys.exit")
    @patch("builtins.print")
    def test_die_default_exit_code(self, mock_print, mock_exit):
        """Test die function with default exit code."""
        die("Error message")

        mock_print.assert_called_once_with("Error message", file=sys.stderr)
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_die_custom_exit_code(self, mock_print, mock_exit):
        """Test die function with custom exit code."""
        die("Custom error", code=42)

        mock_print.assert_called_once_with("Custom error", file=sys.stderr)
        mock_exit.assert_called_once_with(42)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_die_empty_message(self, mock_print, mock_exit):
        """Test die function with empty message."""
        die("")

        mock_print.assert_called_once_with("", file=sys.stderr)
        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_die_multiline_message(self, mock_print, mock_exit):
        """Test die function with multiline message."""
        message = "Error occurred:\nLine 1\nLine 2"
        die(message, code=2)

        mock_print.assert_called_once_with(message, file=sys.stderr)
        mock_exit.assert_called_once_with(2)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_die_zero_exit_code(self, mock_print, mock_exit):
        """Test die function with zero exit code."""
        die("Success message", code=0)

        mock_print.assert_called_once_with("Success message", file=sys.stderr)
        mock_exit.assert_called_once_with(0)


class TestCliIntegration:
    """Integration tests for CLI utility functions."""

    @patch("builtins.input")
    @patch("subprocess.check_output")
    def test_confirmation_and_command_execution(self, mock_check_output, mock_input):
        """Test integration of confirmation and command execution."""
        mock_input.return_value = "y"
        mock_check_output.return_value = "success\n"

        # Simulate a workflow where user confirms and then command runs
        if get_user_confirmation("Run command?"):
            result = sh(["echo", "success"])
            assert result == "success"

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("sys.exit")
    def test_confirmation_denial_and_die(self, mock_exit, mock_print, mock_input):
        """Test workflow where user denies confirmation and program exits."""
        mock_input.return_value = "n"

        # Simulate a workflow where user denies and program exits
        if not get_user_confirmation("Proceed?"):
            die("Operation cancelled")

        # Verify the die function was called
        mock_exit.assert_called_once_with(1)
