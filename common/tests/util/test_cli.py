"""Tests for metta.common.util.cli module."""

import subprocess
import sys
import time
from unittest.mock import Mock, patch

import pytest

from metta.common.util.cli import Spinner, get_user_confirmation, sh, spinner, die
from metta.common.util.text_styles import yellow


class TestSpinner:
    """Test cases for the Spinner class."""

    def test_spinner_init_default_params(self):
        """Test Spinner initialization with default parameters."""
        spin = Spinner()

        assert spin.message == "Processing"
        assert spin.spinner_chars == ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        assert spin.style == yellow
        assert spin._thread is None
        assert spin._spinner_index == 0

    def test_spinner_init_custom_params(self):
        """Test Spinner initialization with custom parameters."""
        custom_chars = [".", "o", "O"]
        custom_style = lambda x: f"[{x}]"

        spin = Spinner("Custom message", custom_chars, custom_style)

        assert spin.message == "Custom message"
        assert spin.spinner_chars == custom_chars
        assert spin.style == custom_style

    def test_spinner_start_stop(self):
        """Test starting and stopping the spinner."""
        with patch('sys.stdout'):
            spin = Spinner("Test", style=lambda x: x)

            spin.start()
            assert spin._thread is not None
            assert spin._thread.is_alive()
            assert spin._thread.daemon is True

            spin.stop()
            time.sleep(0.2)
            assert not spin._thread.is_alive()

    def test_spinner_update_message(self):
        """Test updating spinner message."""
        spin = Spinner("Initial")
        spin.update_message("Updated")
        assert spin.message == "Updated"

    def test_spinner_start_multiple_times(self):
        """Test that starting an already running spinner is safe."""
        with patch('sys.stdout'):
            spin = Spinner("Test", style=lambda x: x)

            spin.start()
            first_thread = spin._thread

            spin.start()  # Should not create new thread
            assert spin._thread is first_thread

            spin.stop()

    def test_spinner_stop_when_not_running(self):
        """Test that stopping a non-running spinner is safe."""
        spin = Spinner("Test")
        spin.stop()  # Should not raise error


class TestSpinnerContextManager:
    """Test cases for the spinner context manager."""

    def test_spinner_context_manager_basic(self):
        """Test basic spinner context manager usage."""
        with patch('sys.stdout'):
            with spinner("Test message") as sp:
                assert isinstance(sp, Spinner)
                assert sp.message == "Test message"
                assert sp._thread is not None
                assert sp._thread.is_alive()

            time.sleep(0.1)
            assert not sp._thread.is_alive()

    def test_spinner_context_manager_with_exception(self):
        """Test that spinner stops even when exception occurs."""
        with patch('sys.stdout'):
            with pytest.raises(ValueError):
                with spinner("Test") as sp:
                    assert sp._thread.is_alive()
                    raise ValueError("Test exception")

            time.sleep(0.1)
            assert not sp._thread.is_alive()

    def test_spinner_context_manager_custom_params(self):
        """Test spinner context manager with custom parameters."""
        custom_chars = ["1", "2", "3"]
        custom_style = lambda x: f"<{x}>"

        with patch('sys.stdout'):
            with spinner("Custom", custom_chars, custom_style) as sp:
                assert sp.spinner_chars == custom_chars
                assert sp.style == custom_style


class TestSh:
    """Test cases for the sh function."""

    @patch('subprocess.check_output')
    def test_sh_success(self, mock_check_output):
        """Test successful command execution with sh."""
        mock_check_output.return_value = "Command output\n"

        result = sh(["echo", "hello"])

        assert result == "Command output"
        mock_check_output.assert_called_once_with(
            ["echo", "hello"],
            text=True
        )

    @patch('subprocess.check_output')
    def test_sh_with_spinner(self, mock_check_output):
        """Test sh with spinner enabled."""
        mock_check_output.return_value = "Output\n"

        with patch('sys.stdout'):  # Suppress spinner output
            result = sh(["echo", "test"], show_spinner=True)

        assert result == "Output"
        mock_check_output.assert_called_once_with(
            ["echo", "test"],
            text=True
        )

    @patch('subprocess.check_output')
    def test_sh_with_custom_spinner_message(self, mock_check_output):
        """Test sh with custom spinner message."""
        mock_check_output.return_value = "Output\n"

        with patch('sys.stdout'):
            result = sh(["long", "command"], show_spinner=True, spinner_message="Custom message")

        assert result == "Output"

    @patch('subprocess.check_output')
    def test_sh_with_kwargs(self, mock_check_output):
        """Test sh with additional kwargs."""
        mock_check_output.return_value = "Output\n"

        sh(["ls"], cwd="/tmp")

        mock_check_output.assert_called_once_with(
            ["ls"],
            text=True,
            cwd="/tmp"
        )

    @patch('subprocess.check_output')
    def test_sh_failure(self, mock_check_output):
        """Test sh when command fails."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "command")

        with pytest.raises(subprocess.CalledProcessError):
            sh(["failing_command"])


class TestGetUserConfirmation:
    """Test cases for the get_user_confirmation function."""

    @patch('builtins.input')
    def test_get_user_confirmation_yes(self, mock_input):
        """Test user confirmation with yes response."""
        mock_input.return_value = "y"

        result = get_user_confirmation("Proceed?")

        assert result is True
        mock_input.assert_called_once_with("Proceed? (Y/n): ")

    @patch('builtins.input')
    def test_get_user_confirmation_empty(self, mock_input):
        """Test user confirmation with empty response (default yes)."""
        mock_input.return_value = ""

        result = get_user_confirmation()

        assert result is True

    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_user_confirmation_no(self, mock_print, mock_input):
        """Test user confirmation with no response."""
        mock_input.return_value = "n"

        result = get_user_confirmation("Proceed?")

        assert result is False
        mock_print.assert_called_once()

    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_user_confirmation_invalid(self, mock_print, mock_input):
        """Test user confirmation with invalid response."""
        mock_input.return_value = "maybe"

        result = get_user_confirmation()

        assert result is False
        mock_print.assert_called_once()


class TestDie:
    """Test cases for the die function."""

    @patch('sys.exit')
    @patch('builtins.print')
    def test_die_default_code(self, mock_print, mock_exit):
        """Test die with default exit code."""
        die("Error message")

        mock_print.assert_called_once_with("Error message", file=sys.stderr)
        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_die_custom_code(self, mock_print, mock_exit):
        """Test die with custom exit code."""
        die("Custom error", code=42)

        mock_print.assert_called_once_with("Custom error", file=sys.stderr)
        mock_exit.assert_called_once_with(42)


class TestCliIntegration:
    """Integration tests for CLI utilities."""

    def test_realistic_command_execution(self):
        """Test command execution with a realistic command."""
        result = sh(["python", "-c", "print('hello world')"])

        assert result == "hello world"

    def test_command_execution_failure_realistic(self):
        """Test command execution with a command that fails."""
        with pytest.raises(subprocess.CalledProcessError):
            sh(["python", "-c", "import sys; sys.exit(1)"])

    def test_spinner_with_real_command(self):
        """Test spinner with real command execution."""
        with patch('sys.stdout'):  # Suppress spinner output
            result = sh(["echo", "test"], show_spinner=True, spinner_message="Running test")

        assert result == "test"
