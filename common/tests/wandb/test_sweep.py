"""Tests for wandb sweep utilities."""

import logging
import time
from unittest.mock import Mock, patch

import pytest

from metta.common.util.retry import retry_on_exception
from metta.sweep.sweep_wandb import sweep_id_from_name


class TestRetryDecorator:
    """Test the retry_on_exception decorator."""

    def test_retry_on_exception_success_first_try(self):
        """Test that function succeeds on first try without retries."""
        mock_func = Mock(return_value="success")

        @retry_on_exception(max_retries=3, retry_delay=0.1)
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_exception_success_after_retries(self):
        """Test that function succeeds after some retries."""
        mock_func = Mock(side_effect=[Exception("fail 1"), Exception("fail 2"), "success"])

        @retry_on_exception(max_retries=3, retry_delay=0.1)
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_on_exception_all_retries_fail(self):
        """Test that exception is raised after all retries fail."""
        mock_func = Mock(side_effect=Exception("always fails"))

        @retry_on_exception(max_retries=3, retry_delay=0.1)
        def test_func():
            return mock_func()

        with pytest.raises(Exception, match="always fails"):
            test_func()

        assert mock_func.call_count == 3

    def test_retry_on_exception_with_specific_exceptions(self):
        """Test that only specific exceptions trigger retries."""
        mock_func = Mock(side_effect=[ValueError("retry this"), RuntimeError("don't retry")])

        @retry_on_exception(max_retries=3, retry_delay=0.1, exceptions=(ValueError,))
        def test_func():
            return mock_func()

        # Should retry on ValueError but not on RuntimeError
        with pytest.raises(RuntimeError, match="don't retry"):
            test_func()

        assert mock_func.call_count == 2  # First ValueError, then RuntimeError

    def test_retry_on_exception_with_logger(self, caplog):
        """Test that retry attempts are logged correctly."""
        mock_func = Mock(side_effect=[Exception("fail 1"), Exception("fail 2"), "success"])
        logger = logging.getLogger("test_logger")

        @retry_on_exception(max_retries=3, retry_delay=0.1, logger=logger)
        def test_func():
            return mock_func()

        with caplog.at_level(logging.INFO):
            result = test_func()

        assert result == "success"
        assert "test_func failed (attempt 1/3): fail 1" in caplog.text
        assert "test_func failed (attempt 2/3): fail 2" in caplog.text
        assert "Retrying in 0.1 seconds..." in caplog.text

    def test_retry_on_exception_preserves_function_attributes(self):
        """Test that decorator preserves function name and docstring."""

        @retry_on_exception(max_retries=3, retry_delay=0.1)
        def test_func():
            """Test function docstring."""
            return "result"

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."

    def test_retry_delay_timing(self):
        """Test that retry delays are applied correctly."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])
        retry_delay = 0.2

        @retry_on_exception(max_retries=2, retry_delay=retry_delay)
        def test_func():
            return mock_func()

        start_time = time.time()
        result = test_func()
        elapsed_time = time.time() - start_time

        assert result == "success"
        assert elapsed_time >= retry_delay
        assert elapsed_time < retry_delay + 0.1  # Allow some margin


class TestSweepIdFromName:
    """Test the sweep_id_from_name function."""

    @patch("wandb.Api")
    def test_sweep_id_from_name_found(self, mock_api_class):
        """Test successfully finding a sweep by name."""
        # Mock the API and project
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_project = Mock()
        mock_api.project.return_value = mock_project

        # Mock sweeps
        mock_sweep = Mock()
        mock_sweep.name = "test_sweep"
        mock_sweep.id = "sweep123"

        mock_project.sweeps.return_value = [
            Mock(name="other_sweep", id="other123"),
            mock_sweep,
            Mock(name="another_sweep", id="another123"),
        ]

        # Call the function
        result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result == "sweep123"
        mock_api.project.assert_called_once_with("test_project", "test_entity")

    @patch("wandb.Api")
    def test_sweep_id_from_name_not_found(self, mock_api_class):
        """Test when sweep is not found."""
        # Mock the API and project
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_project = Mock()
        mock_api.project.return_value = mock_project

        # Mock sweeps without the target sweep
        mock_project.sweeps.return_value = [
            Mock(name="other_sweep", id="other123"),
            Mock(name="another_sweep", id="another123"),
        ]

        # Call the function
        result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result is None

    @patch("wandb.Api")
    def test_sweep_id_from_name_project_not_found(self, mock_api_class):
        """Test when project doesn't exist."""
        # Mock the API
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Make project access fail
        mock_api.project.side_effect = Exception("Project not found")

        # Call the function
        result = sweep_id_from_name("nonexistent_project", "test_entity", "test_sweep")

        assert result is None

    @patch("wandb.Api")
    def test_sweep_id_from_name_with_network_retry(self, mock_api_class):
        """Test that network errors trigger retries."""
        # This test verifies that the retry decorator works by mocking
        # the entire API to fail and then succeed

        # We'll simulate the retry behavior by having the API constructor
        # fail initially, then succeed
        call_count = 0

        def api_constructor_side_effect():
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise Exception(f"Network error {call_count}")

            # On the third call, return a successful mock
            mock_api = Mock()
            mock_project = Mock()
            mock_api.project.return_value = mock_project

            mock_sweep = Mock()
            mock_sweep.name = "test_sweep"
            mock_sweep.id = "sweep123"
            mock_project.sweeps.return_value = [mock_sweep]

            return mock_api

        mock_api_class.side_effect = api_constructor_side_effect

        # Call the function with minimal retry delay by patching sleep
        with patch("time.sleep"):
            result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        # Should succeed after retries
        assert result == "sweep123"
        assert call_count == 3  # Should have been called 3 times

    @patch("wandb.Api")
    def test_sweep_id_from_name_all_retries_fail(self, mock_api_class):
        """Test that function returns None when all retries fail."""
        # Mock the API to always fail
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_api.project.side_effect = Exception("Persistent network error")

        # Call the function with minimal retry delay
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result is None
