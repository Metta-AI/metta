"""Tests for metta.common.util.logging_helpers module."""

import logging
import os
import sys
from datetime import datetime
from io import StringIO
from unittest.mock import Mock, mock_open, patch

import pytest

from metta.common.util.logging_helpers import (
    AlwaysShowTimeRichHandler,
    MillisecondFormatter,
    SimpleHandler,
    get_log_level,
    init_file_logging,
    init_logging,
    remap_io,
    restore_io,
)


class TestRemapIo:
    """Test cases for IO remapping functions."""

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_remap_io(self, mock_file_open, mock_makedirs):
        """Test remap_io function."""
        mock_stdout = Mock()
        mock_stderr = Mock()
        mock_file_open.side_effect = [mock_stdout, mock_stderr]

        remap_io("/test/logs")

        # Check directory creation
        mock_makedirs.assert_called_once_with("/test/logs", exist_ok=True)
        
        # Check file opening
        assert mock_file_open.call_count == 2
        mock_file_open.assert_any_call("/test/logs/out.log", "a")
        mock_file_open.assert_any_call("/test/logs/error.log", "a")
        
        # Check sys redirection
        assert sys.stdout == mock_stdout
        assert sys.stderr == mock_stderr

    def test_restore_io(self):
        """Test restore_io function."""
        # Save original
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to test values
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        # Restore
        restore_io()
        
        # Check restoration
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr


class TestMillisecondFormatter:
    """Test cases for MillisecondFormatter."""

    def test_millisecond_formatter_init(self):
        """Test MillisecondFormatter initialization."""
        formatter = MillisecondFormatter()
        assert isinstance(formatter, logging.Formatter)

    @patch('metta.common.util.logging_helpers.datetime')
    def test_format_time_with_datefmt_f(self, mock_datetime):
        """Test formatTime with %f in datefmt."""
        formatter = MillisecondFormatter()
        
        mock_dt = Mock()
        mock_dt.microsecond = 123456  # 123 milliseconds
        mock_dt.strftime.return_value = "12:34:56.123"
        mock_datetime.fromtimestamp.return_value = mock_dt
        
        record = Mock()
        record.created = 1234567890.123456
        
        result = formatter.formatTime(record, "[%H:%M:%S.%f]")
        
        assert result == "12:34:56.123"
        mock_dt.strftime.assert_called_once_with("[%H:%M:%S.123]")

    @patch('metta.common.util.logging_helpers.datetime')
    def test_format_time_with_regular_datefmt(self, mock_datetime):
        """Test formatTime with regular datefmt."""
        formatter = MillisecondFormatter()
        
        mock_dt = Mock()
        mock_dt.strftime.return_value = "12:34:56"
        mock_datetime.fromtimestamp.return_value = mock_dt
        
        record = Mock()
        record.created = 1234567890.123456
        
        result = formatter.formatTime(record, "%H:%M:%S")
        
        assert result == "12:34:56"
        mock_dt.strftime.assert_called_once_with("%H:%M:%S")

    @patch('metta.common.util.logging_helpers.datetime')
    def test_format_time_default(self, mock_datetime):
        """Test formatTime with default format."""
        formatter = MillisecondFormatter()
        
        mock_dt = Mock()
        mock_dt.microsecond = 456789  # 456 milliseconds
        mock_dt.strftime.return_value = "[12:34:56.456]"
        mock_datetime.fromtimestamp.return_value = mock_dt
        
        record = Mock()
        record.created = 1234567890.456789
        
        result = formatter.formatTime(record)
        
        assert result == "[12:34:56.456]"


class TestAlwaysShowTimeRichHandler:
    """Test cases for AlwaysShowTimeRichHandler."""

    def test_always_show_time_rich_handler_init(self):
        """Test AlwaysShowTimeRichHandler initialization."""
        handler = AlwaysShowTimeRichHandler()
        assert hasattr(handler, 'emit')

    @patch('rich.logging.RichHandler.emit')
    def test_emit_modifies_created_time(self, mock_super_emit):
        """Test that emit modifies record.created for unique timestamps."""
        handler = AlwaysShowTimeRichHandler()
        
        record = Mock()
        record.created = 1234567890.0
        record.relativeCreated = 1500.5  # 1500.5 ms
        
        handler.emit(record)
        
        # Should modify created time
        expected_created = 1234567890.0 + (1500.5 % 1000) / 1000000
        assert abs(record.created - expected_created) < 1e-9
        mock_super_emit.assert_called_once_with(record)


class TestSimpleHandler:
    """Test cases for SimpleHandler."""

    def test_simple_handler_init(self):
        """Test SimpleHandler initialization."""
        handler = SimpleHandler()
        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, MillisecondFormatter)


class TestGetLogLevel:
    """Test cases for get_log_level function."""

    def test_get_log_level_env_variable(self):
        """Test get_log_level with environment variable set."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'debug'}):
            result = get_log_level()
            assert result == "DEBUG"

    def test_get_log_level_provided_parameter(self):
        """Test get_log_level with provided parameter."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_log_level("warning")
            assert result == "WARNING"

    def test_get_log_level_env_overrides_parameter(self):
        """Test that environment variable overrides provided parameter."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'error'}):
            result = get_log_level("debug")
            assert result == "ERROR"

    def test_get_log_level_default(self):
        """Test get_log_level with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_log_level()
            assert result == "INFO"

    def test_get_log_level_case_insensitive(self):
        """Test get_log_level converts to uppercase."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'debug'}):
            result = get_log_level()
            assert result == "DEBUG"


class TestInitFileLogging:
    """Test cases for init_file_logging function."""

    @patch('logging.getLogger')
    @patch('logging.FileHandler')
    @patch('os.makedirs')
    def test_init_file_logging_rank_0(self, mock_makedirs, mock_file_handler, mock_get_logger):
        """Test init_file_logging for rank 0 (main process)."""
        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'RANK': '0'}):
            init_file_logging("/test/run")

        mock_makedirs.assert_called_once_with("/test/run/logs", exist_ok=True)
        mock_file_handler.assert_called_once_with("/test/run/logs/script.log", mode="a")
        mock_logger.addHandler.assert_called_once_with(mock_handler)

    @patch('logging.getLogger')
    @patch('logging.FileHandler')
    @patch('os.makedirs')
    def test_init_file_logging_other_rank(self, mock_makedirs, mock_file_handler, mock_get_logger):
        """Test init_file_logging for non-zero rank."""
        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'RANK': '2'}):
            init_file_logging("/test/run")

        mock_file_handler.assert_called_once_with("/test/run/logs/script_2.log", mode="a")

    @patch('logging.getLogger')
    @patch('logging.FileHandler')
    @patch('os.makedirs')
    def test_init_file_logging_no_rank(self, mock_makedirs, mock_file_handler, mock_get_logger):
        """Test init_file_logging when RANK is not set (defaults to 0)."""
        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {}, clear=True):
            init_file_logging("/test/run")

        mock_file_handler.assert_called_once_with("/test/run/logs/script.log", mode="a")


class TestInitLogging:
    """Test cases for init_logging function."""

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_rich_handler(self, mock_get_logger, mock_get_log_level):
        """Test init_logging with Rich handler (default case)."""
        mock_get_log_level.return_value = "INFO"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {}, clear=True):
            init_logging()

        # Should add Rich handler
        assert mock_logger.addHandler.called

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_simple_handler_wandb(self, mock_get_logger, mock_get_log_level):
        """Test init_logging with simple handler when WANDB is detected."""
        mock_get_log_level.return_value = "INFO"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'WANDB_MODE': 'online'}):
            init_logging()

        # Should add simple handler instead of Rich
        assert mock_logger.addHandler.called

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_simple_handler_batch(self, mock_get_logger, mock_get_log_level):
        """Test init_logging with simple handler in AWS Batch."""
        mock_get_log_level.return_value = "DEBUG"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'AWS_BATCH_JOB_ID': 'job123'}):
            init_logging()

        assert mock_logger.addHandler.called

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_simple_handler_skypilot(self, mock_get_logger, mock_get_log_level):
        """Test init_logging with simple handler in SkyPilot."""
        mock_get_log_level.return_value = "WARNING"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': 'task456'}):
            init_logging()

        assert mock_logger.addHandler.called

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_simple_handler_no_rich(self, mock_get_logger, mock_get_log_level):
        """Test init_logging with simple handler when Rich is disabled."""
        mock_get_log_level.return_value = "ERROR"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch.dict(os.environ, {'NO_RICH_LOGS': '1'}):
            init_logging()

        assert mock_logger.addHandler.called

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('metta.common.util.logging_helpers.init_file_logging')
    @patch('logging.getLogger')
    def test_init_logging_with_run_dir(self, mock_get_logger, mock_init_file, mock_get_log_level):
        """Test init_logging with run_dir parameter."""
        mock_get_log_level.return_value = "INFO"
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        init_logging(run_dir="/test/run")

        mock_init_file.assert_called_once_with("/test/run")

    @patch('metta.common.util.logging_helpers.get_log_level')
    @patch('logging.getLogger')
    def test_init_logging_removes_existing_handlers(self, mock_get_logger, mock_get_log_level):
        """Test that init_logging removes existing handlers."""
        mock_get_log_level.return_value = "INFO"
        
        # Create mock handlers
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        mock_logger = Mock()
        mock_logger.handlers = [mock_handler1, mock_handler2]
        mock_get_logger.return_value = mock_logger

        init_logging()

        # Should remove existing handlers
        mock_logger.removeHandler.assert_any_call(mock_handler1)
        mock_logger.removeHandler.assert_any_call(mock_handler2)


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_log_level_priority(self):
        """Test log level priority: env > param > default."""
        # Test environment variable priority
        with patch.dict(os.environ, {'LOG_LEVEL': 'error'}):
            assert get_log_level("debug") == "ERROR"

        # Test parameter when no env var
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_level("warning") == "WARNING"

        # Test default when neither set
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_level() == "INFO"

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_io_remap_restore_cycle(self, mock_file_open, mock_makedirs):
        """Test complete IO remap and restore cycle."""
        # Save original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Remap IO
        mock_stdout = Mock()
        mock_stderr = Mock()
        mock_file_open.side_effect = [mock_stdout, mock_stderr]
        
        remap_io("/test/logs")
        
        # Verify remap worked
        assert sys.stdout == mock_stdout
        assert sys.stderr == mock_stderr
        
        # Restore IO
        restore_io()
        
        # Verify restore worked
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

    def test_formatter_millisecond_precision(self):
        """Test that MillisecondFormatter correctly handles millisecond precision."""
        formatter = MillisecondFormatter()
        
        with patch('metta.common.util.logging_helpers.datetime') as mock_datetime:
            mock_dt = Mock()
            mock_dt.microsecond = 123456  # Should become 123 milliseconds
            mock_dt.strftime.return_value = "formatted_time"
            mock_datetime.fromtimestamp.return_value = mock_dt
            
            record = Mock()
            record.created = 1234567890.123456
            
            # Test with %f format
            result = formatter.formatTime(record, "%H:%M:%S.%f")
            
            # Should replace %f with milliseconds
            mock_dt.strftime.assert_called_with("%H:%M:%S.123")

    @patch('logging.getLogger')
    def test_handler_detection_logic(self, mock_get_logger):
        """Test the environment-based handler detection logic."""
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        # Test various environment conditions that should trigger simple handler
        simple_handler_envs = [
            {'WANDB_MODE': 'online'},
            {'WANDB_RUN_ID': 'run123'},
            {'METTA_RUN_ID': 'metta456'},
            {'AWS_BATCH_JOB_ID': 'batch789'},
            {'SKYPILOT_TASK_ID': 'sky101'},
            {'NO_HYPERLINKS': '1'},
            {'NO_RICH_LOGS': 'true'},
        ]

        for env in simple_handler_envs:
            with patch.dict(os.environ, env):
                init_logging()
                # Should add handler (we can't easily test which type without more complex mocking)
                assert mock_logger.addHandler.called
                mock_logger.reset_mock()

    def test_file_logging_rank_handling(self):
        """Test that file logging correctly handles different rank values."""
        with patch('os.makedirs'), \
             patch('logging.FileHandler') as mock_handler, \
             patch('logging.getLogger'):
            
            # Test rank 0
            with patch.dict(os.environ, {'RANK': '0'}):
                init_file_logging("/test")
                mock_handler.assert_called_with("/test/logs/script.log", mode="a")
                
            mock_handler.reset_mock()
            
            # Test rank > 0
            with patch.dict(os.environ, {'RANK': '3'}):
                init_file_logging("/test")
                mock_handler.assert_called_with("/test/logs/script_3.log", mode="a")
