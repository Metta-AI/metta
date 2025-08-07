"""Tests for metta.common.util.heartbeat module."""

import json
import os
import signal
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from metta.common.util.heartbeat import (
    WANDB_IPC_FILENAME,
    _send_wandb_alert_with_timeout,
    _main,
    monitor_heartbeat,
    record_heartbeat,
)


class TestRecordHeartbeat:
    """Test cases for record_heartbeat function."""

    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('time.time')
    def test_record_heartbeat_success(self, mock_time, mock_open, mock_makedirs):
        """Test successful heartbeat recording."""
        mock_time.return_value = 1234567890.123
        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        with patch.dict(os.environ, {'HEARTBEAT_FILE': '/tmp/heartbeat.txt'}):
            record_heartbeat()

        mock_makedirs.assert_called_once_with('/tmp', exist_ok=True)
        mock_open.assert_called_once_with('/tmp/heartbeat.txt', 'w')
        mock_file.write.assert_called_once_with('1234567890.123')

    def test_record_heartbeat_no_env_var(self):
        """Test record_heartbeat when HEARTBEAT_FILE is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise any errors
            record_heartbeat()

    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('metta.common.util.heartbeat.logger')
    def test_record_heartbeat_write_failure(self, mock_logger, mock_open, mock_makedirs):
        """Test record_heartbeat when file write fails."""
        mock_open.side_effect = OSError("Permission denied")

        with patch.dict(os.environ, {'HEARTBEAT_FILE': '/tmp/heartbeat.txt'}):
            record_heartbeat()

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Failed to write heartbeat" in warning_msg

    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('metta.common.util.heartbeat.logger')
    def test_record_heartbeat_makedirs_failure(self, mock_logger, mock_open, mock_makedirs):
        """Test record_heartbeat when directory creation fails."""
        mock_makedirs.side_effect = OSError("Permission denied")

        with patch.dict(os.environ, {'HEARTBEAT_FILE': '/tmp/subdir/heartbeat.txt'}):
            record_heartbeat()

        mock_logger.warning.assert_called_once()


class TestSendWandbAlertWithTimeout:
    """Test cases for _send_wandb_alert_with_timeout function."""

    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_no_ipc_file_path(self, mock_logger):
        """Test _send_wandb_alert_with_timeout when no IPC file path provided."""
        _send_wandb_alert_with_timeout("Test Alert", None)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "W&B IPC file path not provided" in warning_msg
        assert "Test Alert" in warning_msg

    @patch('builtins.open')
    @patch('wandb.Api')
    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_success(self, mock_logger, mock_wandb_api, mock_open):
        """Test successful W&B alert sending."""
        # Mock IPC file content
        ipc_data = {
            "run_id": "test_run_123",
            "project": "test_project",
            "entity": "test_entity"
        }
        mock_file = Mock()
        mock_file.read.return_value = json.dumps(ipc_data)
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        # Mock W&B API
        mock_api_instance = Mock()
        mock_run = Mock()
        mock_api_instance.run.return_value = mock_run
        mock_wandb_api.return_value = mock_api_instance

        _send_wandb_alert_with_timeout("Test Alert", "/tmp/wandb_ipc.json")

        mock_open.assert_called_once_with("/tmp/wandb_ipc.json", "r")
        mock_wandb_api.assert_called_once()
        mock_api_instance.run.assert_called_once_with("test_entity/test_project/test_run_123")
        mock_run.alert.assert_called_once_with(
            title="Test Alert",
            text="Test Alert",
            level="WARN"
        )
        mock_logger.info.assert_called_once_with("Sent W&B alert: 'Test Alert'")

    @patch('builtins.open')
    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_file_not_found(self, mock_logger, mock_open):
        """Test _send_wandb_alert_with_timeout when IPC file doesn't exist."""
        mock_open.side_effect = FileNotFoundError("No such file")

        _send_wandb_alert_with_timeout("Test Alert", "/tmp/missing.json")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "IPC file not found" in warning_msg

    @patch('builtins.open')
    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_invalid_json(self, mock_logger, mock_open):
        """Test _send_wandb_alert_with_timeout with invalid JSON in IPC file."""
        mock_file = Mock()
        mock_file.read.return_value = "invalid json"
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        _send_wandb_alert_with_timeout("Test Alert", "/tmp/wandb_ipc.json")

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Failed to parse IPC file" in warning_msg

    @patch('builtins.open')
    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_missing_required_fields(self, mock_logger, mock_open):
        """Test _send_wandb_alert_with_timeout with missing required fields in IPC."""
        # Missing run_id
        ipc_data = {
            "project": "test_project",
            "entity": "test_entity"
        }
        mock_file = Mock()
        mock_file.read.return_value = json.dumps(ipc_data)
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        _send_wandb_alert_with_timeout("Test Alert", "/tmp/wandb_ipc.json")

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Missing required W&B info" in warning_msg

    @patch('builtins.open')
    @patch('wandb.Api')
    @patch('metta.common.util.heartbeat.logger')
    def test_send_wandb_alert_api_failure(self, mock_logger, mock_wandb_api, mock_open):
        """Test _send_wandb_alert_with_timeout when W&B API call fails."""
        # Mock IPC file content
        ipc_data = {
            "run_id": "test_run_123",
            "project": "test_project",
            "entity": "test_entity"
        }
        mock_file = Mock()
        mock_file.read.return_value = json.dumps(ipc_data)
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        # Mock W&B API failure
        mock_api_instance = Mock()
        mock_api_instance.run.side_effect = Exception("API Error")
        mock_wandb_api.return_value = mock_api_instance

        _send_wandb_alert_with_timeout("Test Alert", "/tmp/wandb_ipc.json")

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Failed to send W&B alert" in warning_msg


class TestMonitorHeartbeat:
    """Test cases for monitor_heartbeat function."""

    @patch('time.sleep')
    @patch('os.path.getmtime')
    @patch('time.time')
    def test_monitor_heartbeat_file_exists_recent(self, mock_time, mock_getmtime, mock_sleep):
        """Test monitor_heartbeat when file exists and is recent."""
        mock_time.return_value = 1000.0
        mock_getmtime.return_value = 950.0  # 50 seconds ago
        
        # Stop after one iteration
        mock_sleep.side_effect = KeyboardInterrupt()
        
        try:
            monitor_heartbeat("/tmp/heartbeat.txt", pid=12345, timeout=600.0, check_interval=60.0)
        except KeyboardInterrupt:
            pass
        
        mock_getmtime.assert_called_with("/tmp/heartbeat.txt")
        mock_sleep.assert_called_with(60.0)

    @patch('time.sleep')
    @patch('os.path.getmtime')
    @patch('time.time')
    @patch('os.killpg')
    @patch('metta.common.util.heartbeat._send_wandb_alert_with_timeout')
    def test_monitor_heartbeat_timeout_triggers_kill(self, mock_alert, mock_killpg, mock_time, mock_getmtime, mock_sleep):
        """Test monitor_heartbeat kills process when timeout exceeded."""
        mock_time.return_value = 1000.0
        mock_getmtime.return_value = 100.0  # 900 seconds ago (exceeds 600s timeout)
        
        # Should break out of loop after killing
        monitor_heartbeat("/tmp/heartbeat.txt", pid=12345, timeout=600.0, check_interval=60.0)
        
        # Should send alert and kill process
        mock_alert.assert_called_once()
        assert mock_killpg.call_count >= 1  # SIGTERM and potentially SIGKILL

    @patch('time.sleep')
    @patch('os.path.getmtime')
    @patch('time.time')
    def test_monitor_heartbeat_file_not_found(self, mock_time, mock_getmtime, mock_sleep):
        """Test monitor_heartbeat when heartbeat file doesn't exist."""
        mock_time.return_value = 1000.0
        mock_getmtime.side_effect = FileNotFoundError()
        
        # Stop after one iteration
        mock_sleep.side_effect = KeyboardInterrupt()
        
        try:
            monitor_heartbeat("/tmp/nonexistent.txt", pid=12345, timeout=600.0, check_interval=60.0)
        except KeyboardInterrupt:
            pass
        
        # Should handle FileNotFoundError gracefully
        mock_getmtime.assert_called_with("/tmp/nonexistent.txt")


class TestMainFunction:
    """Test cases for the _main function."""

    @patch('metta.common.util.heartbeat.record_heartbeat')
    @patch('os.makedirs')
    def test_main_heartbeat_command(self, mock_makedirs, mock_record):
        """Test _main function with heartbeat command."""
        _main(['heartbeat', '/tmp/heartbeat.txt'])
        
        mock_makedirs.assert_called_once_with('/tmp', exist_ok=True)
        mock_record.assert_called_once()

    @patch('metta.common.util.heartbeat.monitor_heartbeat')
    def test_main_monitor_command_default_args(self, mock_monitor):
        """Test _main function with monitor command using default arguments."""
        with patch('os.getpid', return_value=12345):
            _main(['monitor', '/tmp/heartbeat.txt'])
        
        mock_monitor.assert_called_once_with(
            '/tmp/heartbeat.txt',
            pid=12345,
            timeout=600.0,
            check_interval=60.0
        )

    @patch('metta.common.util.heartbeat.monitor_heartbeat')
    def test_main_monitor_command_custom_args(self, mock_monitor):
        """Test _main function with monitor command using custom arguments."""
        _main(['monitor', '/tmp/heartbeat.txt', '--pid', '9999', '--timeout', '300', '--interval', '30'])
        
        mock_monitor.assert_called_once_with(
            '/tmp/heartbeat.txt',
            pid=9999,
            timeout=300.0,
            check_interval=30.0
        )




class TestIntegration:
    """Integration tests for heartbeat functionality."""

    def test_record_heartbeat_integration(self):
        """Test record_heartbeat with real file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            heartbeat_file = os.path.join(temp_dir, "test_heartbeat.txt")
            
            with patch.dict(os.environ, {'HEARTBEAT_FILE': heartbeat_file}):
                with patch('time.time', return_value=1234567890.5):
                    record_heartbeat()
            
            # Verify file was created and contains correct timestamp
            assert os.path.exists(heartbeat_file)
            with open(heartbeat_file, 'r') as f:
                content = f.read()
                assert content == "1234567890.5"

    def test_wandb_alert_integration(self):
        """Test _send_wandb_alert_with_timeout integration with file system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ipc_file = os.path.join(temp_dir, "wandb_ipc.json")
            
            # Create IPC file with test data
            ipc_data = {
                "run_id": "test_run_123",
                "project": "test_project",
                "entity": "test_entity"
            }
            with open(ipc_file, 'w') as f:
                json.dump(ipc_data, f)
            
            with patch('wandb.Api') as mock_wandb_api:
                mock_api_instance = Mock()
                mock_run = Mock()
                mock_api_instance.run.return_value = mock_run
                mock_wandb_api.return_value = mock_api_instance
                
                _send_wandb_alert_with_timeout("Integration Test Alert", ipc_file)
                
                # Verify W&B API was called correctly
                mock_wandb_api.assert_called()
                mock_api_instance.run.assert_called_with("test_entity/test_project/test_run_123")

    def test_wandb_ipc_filename_constant(self):
        """Test that WANDB_IPC_FILENAME constant is correctly defined."""
        assert WANDB_IPC_FILENAME == "wandb_ipc.json"

    def test_command_line_interface_integration(self):
        """Test command line interface integration."""
        # Test heartbeat command
        with patch('metta.common.util.heartbeat.record_heartbeat') as mock_record:
            with patch('os.makedirs') as mock_makedirs:
                _main(['heartbeat', '/test/path/heartbeat.txt'])
                mock_makedirs.assert_called_once()
                mock_record.assert_called_once()
        
        # Test monitor command  
        with patch('metta.common.util.heartbeat.monitor_heartbeat') as mock_monitor:
            _main(['monitor', '/test/path/heartbeat.txt', '--pid', '1234'])
            mock_monitor.assert_called_once()
