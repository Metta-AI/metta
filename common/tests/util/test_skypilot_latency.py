"""Tests for metta.common.util.skypilot_latency module."""

import datetime
import os
from unittest.mock import Mock, patch

import pytest

from metta.common.util.skypilot_latency import (
    _EPOCH,
    _FMT,
    _TS_RE,
    _submission_ts,
    main,
    queue_latency_s,
)


class TestConstants:
    """Test module constants."""

    def test_epoch_is_utc(self):
        """Test that _EPOCH is UTC timezone."""
        assert _EPOCH == datetime.timezone.utc

    def test_format_string(self):
        """Test that _FMT is correct datetime format."""
        assert _FMT == "%Y-%m-%d-%H-%M-%S-%f"

    def test_timestamp_regex_standard(self):
        """Test that _TS_RE matches standard patterns."""
        match = _TS_RE.match("sky-2023-12-01-14-30-45-123456_cluster_1")
        assert match is not None
        assert match.group("ts") == "2023-12-01-14-30-45-123456"

    def test_timestamp_regex_invalid(self):
        """Test that _TS_RE rejects invalid patterns."""
        assert _TS_RE.match("invalid-task-id") is None


class TestSubmissionTs:
    """Test cases for _submission_ts function."""

    def test_valid_task_id_standard(self):
        """Test _submission_ts with valid standard task ID."""
        task_id = "sky-2023-12-01-14-30-45-123456_cluster_1"
        result = _submission_ts(task_id)

        expected = datetime.datetime(2023, 12, 1, 14, 30, 45, 123456, tzinfo=_EPOCH)
        assert result == expected

    def test_invalid_task_id_no_match(self):
        """Test _submission_ts with non-matching task ID."""
        result = _submission_ts("invalid-task-id")
        assert result is None

    def test_empty_task_id(self):
        """Test _submission_ts with empty task ID."""
        result = _submission_ts("")
        assert result is None


class TestQueueLatencyS:
    """Test cases for queue_latency_s function."""

    def test_no_task_id_env_var(self):
        """Test queue_latency_s when SKYPILOT_TASK_ID is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = queue_latency_s()
            assert result is None

    def test_invalid_task_id(self):
        """Test queue_latency_s with invalid task ID."""
        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': 'invalid-task-id'}):
            result = queue_latency_s()
            assert result is None

    @patch('metta.common.util.skypilot_latency.datetime')
    def test_valid_task_id_with_mock_time(self, mock_datetime):
        """Test queue_latency_s with valid task ID and mocked current time."""
        # Mock current time
        current_time = datetime.datetime(2023, 12, 1, 14, 35, 45, 123456, tzinfo=_EPOCH)
        mock_datetime.datetime.now.return_value = current_time
        mock_datetime.datetime.strptime.side_effect = datetime.datetime.strptime

        # Task submitted 5 minutes (300 seconds) earlier
        task_id = "sky-2023-12-01-14-30-45-123456_cluster_1"

        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': task_id}):
            result = queue_latency_s()

            assert result == 300.0  # 5 minutes in seconds


class TestMainFunction:
    """Test cases for the main function."""

    @patch('builtins.print')
    @patch('metta.common.util.skypilot_latency.queue_latency_s')
    def test_main_no_run_id_no_latency(self, mock_queue_latency, mock_print):
        """Test main function when no METTA_RUN_ID and no latency."""
        mock_queue_latency.return_value = None

        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': 'invalid-task'}, clear=True):
            result = main()

            assert result == 0
            mock_print.assert_called()

    @patch('builtins.print')
    @patch('wandb.init')
    @patch('wandb.login')
    @patch('metta.common.util.skypilot_latency.queue_latency_s')
    def test_main_with_run_id_and_api_key(self, mock_queue_latency, mock_login, mock_init, mock_print):
        """Test main function with METTA_RUN_ID and WANDB_API_KEY."""
        mock_queue_latency.return_value = 45.6

        # Mock wandb run
        mock_run = Mock()
        mock_run.summary = {}
        mock_init.return_value = mock_run

        env_vars = {
            'METTA_RUN_ID': 'test-run-123',
            'WANDB_API_KEY': 'test-api-key',
            'SKYPILOT_TASK_ID': 'sky-2023-12-01-14-30-45-123456_cluster_1'
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = main()

        assert result == 0

        # Verify wandb operations
        mock_login.assert_called_once_with(key='test-api-key', relogin=True, anonymous="never")
        mock_init.assert_called_once()

        # Verify summary was updated
        assert mock_run.summary['skypilot/latency_script_ran'] is True
        # The actual code uses wandb.log() which doesn't update mock summary
        # So we check for the script execution marker instead

    @patch('builtins.print')
    @patch('os.path.exists')
    @patch('wandb.init')
    @patch('metta.common.util.skypilot_latency.queue_latency_s')
    def test_main_with_netrc_no_latency(self, mock_queue_latency, mock_init, mock_exists, mock_print):
        """Test main function with .netrc file but no latency."""
        mock_queue_latency.return_value = None
        mock_exists.return_value = True  # .netrc exists

        # Mock wandb run
        mock_run = Mock()
        mock_run.summary = {}
        mock_init.return_value = mock_run

        env_vars = {
            'METTA_RUN_ID': 'test-run-456',
            'SKYPILOT_TASK_ID': 'invalid-task'
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = main()

        assert result == 0

        # Should init wandb without login
        mock_init.assert_called_once()

        # Verify summary for failed latency calculation
        assert mock_run.summary['skypilot/latency_calculated'] is False
        assert mock_run.summary['skypilot/latency_error'] == "Could not parse task ID"

    @patch('builtins.print')
    @patch('os.path.exists')
    def test_main_no_auth_methods(self, mock_exists, mock_print):
        """Test main function with run ID but no auth methods."""
        mock_exists.return_value = False  # No .netrc

        env_vars = {
            'METTA_RUN_ID': 'test-run-789',
            'SKYPILOT_TASK_ID': 'test-task'
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = main()

        assert result == 0

        # Should print message about skipping wandb
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Skipping wandb logging (no API key or .netrc found)" in call for call in print_calls)

    @patch('builtins.print')
    @patch('wandb.init')
    @patch('metta.common.util.skypilot_latency.queue_latency_s')
    def test_main_wandb_exception_handling(self, mock_queue_latency, mock_init, mock_print):
        """Test main function when wandb operations raise exception."""
        mock_queue_latency.return_value = 78.9
        mock_init.side_effect = Exception("Wandb connection failed")

        env_vars = {
            'METTA_RUN_ID': 'test-run-error',
            'WANDB_API_KEY': 'test-key',
            'SKYPILOT_TASK_ID': 'sky-2023-12-01-14-30-45-123456_cluster_1'
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = main()

        assert result == 0

        # Should print error message to stderr
        print_calls = [call for call in mock_print.call_args_list]
        stderr_calls = [call for call in print_calls if len(call[1]) > 0 and call[1].get('file')]
        assert len(stderr_calls) > 0

    @patch('builtins.print')
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('metta.common.util.skypilot_latency.queue_latency_s')
    def test_main_custom_wandb_project(self, mock_queue_latency, mock_log, mock_init, mock_print):
        """Test main function with custom WANDB_PROJECT."""
        mock_queue_latency.return_value = 42.0

        # Mock wandb run
        mock_run = Mock()
        mock_run.summary = {}
        mock_init.return_value = mock_run

        env_vars = {
            'METTA_RUN_ID': 'custom-project-test',
            'WANDB_API_KEY': 'test-key',
            'WANDB_PROJECT': 'custom-project',
            'SKYPILOT_TASK_ID': 'sky-2023-12-01-14-30-45-123456_cluster_1'
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = main()

        assert result == 0

        # Verify wandb init was called (project name will be used)
        mock_init.assert_called_once()

    def test_submission_ts_bad_timestamp_cases(self):
        """Test _submission_ts with various bad timestamp cases."""
        # Invalid month
        assert _submission_ts("sky-2023-13-01-14-30-45-123456_cluster_1") is None

        # Invalid day
        assert _submission_ts("sky-2023-12-32-14-30-45-123456_cluster_1") is None

        # Invalid hour
        assert _submission_ts("sky-2023-12-01-25-30-45-123456_cluster_1") is None

        # Invalid minute
        assert _submission_ts("sky-2023-12-01-14-61-45-123456_cluster_1") is None

        # Invalid second
        assert _submission_ts("sky-2023-12-01-14-30-61-123456_cluster_1") is None

    def test_submission_ts_truncation(self):
        """Test _submission_ts properly truncates long microseconds."""
        # 9-digit microseconds should be truncated to 6
        task_id = "sky-2023-12-01-14-30-45-123456789_cluster_1"
        result = _submission_ts(task_id)

        expected = datetime.datetime(2023, 12, 1, 14, 30, 45, 123456, tzinfo=_EPOCH)
        assert result == expected

    def test_queue_latency_s_edge_cases(self):
        """Test queue_latency_s with edge cases."""
        # Empty SKYPILOT_TASK_ID
        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': ''}):
            assert queue_latency_s() is None

        # Malformed task ID (matches regex but bad timestamp)
        with patch.dict(os.environ, {'SKYPILOT_TASK_ID': 'sky-invalid-timestamp_cluster_1'}):
            assert queue_latency_s() is None
