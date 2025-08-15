"""Tests for the retry_on_exception decorator."""

import logging
import time
from unittest.mock import Mock

import pytest

from metta.common.util.retry import retry_function, retry_on_exception


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
        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_retry_on_exception_all_retries_fail(self):
        """Test that exception is raised after all retries fail."""
        mock_func = Mock(side_effect=Exception("always fails"))

        @retry_on_exception(max_retries=3, retry_delay=0.1)
        def test_func():
            return mock_func()

        with pytest.raises(Exception, match="always fails"):
            test_func()

        assert mock_func.call_count == 4  # Initial + 3 retries

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
        assert "test_func failed: fail 1" in caplog.text  # Initial attempt
        assert "test_func failed (retry 1/3): fail 2" in caplog.text  # First retry
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


class TestRetryFunction:
    """Test the retry_function utility."""

    def test_retry_function_success_first_try(self):
        """Test that function succeeds on first try without retries."""
        mock_func = Mock(return_value="success")

        result = retry_function(mock_func, max_retries=3, retry_delay=0.1)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_function_success_after_retries(self):
        """Test that function succeeds after some retries."""
        mock_func = Mock(side_effect=[Exception("fail 1"), Exception("fail 2"), "success"])

        result = retry_function(mock_func, max_retries=3, retry_delay=0.1)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_function_all_retries_fail(self):
        """Test that exception is raised after all retries fail."""
        mock_func = Mock(side_effect=Exception("always fails"))

        with pytest.raises(Exception, match="always fails"):
            retry_function(mock_func, max_retries=3, retry_delay=0.1)

        assert mock_func.call_count == 4  # Initial + 3 retries

    def test_retry_function_with_lambda(self):
        """Test that retry_function works with lambda functions."""
        counter = {"value": 0}

        def failing_func():
            counter["value"] += 1
            if counter["value"] < 3:
                raise ValueError(f"Attempt {counter['value']} failed")
            return "success"

        result = retry_function(lambda: failing_func(), max_retries=3, retry_delay=0.1)

        assert result == "success"
        assert counter["value"] == 3

    def test_retry_function_with_specific_exceptions(self):
        """Test that only specific exceptions trigger retries."""
        mock_func = Mock(side_effect=[ValueError("retry this"), RuntimeError("don't retry")])

        with pytest.raises(RuntimeError, match="don't retry"):
            retry_function(mock_func, max_retries=3, retry_delay=0.1, exceptions=(ValueError,))

        assert mock_func.call_count == 2  # First ValueError, then RuntimeError

    def test_retry_function_with_logger(self, caplog):
        """Test that retry attempts are logged correctly."""
        mock_func = Mock(side_effect=[Exception("fail 1"), Exception("fail 2"), "success"])
        logger = logging.getLogger("test_logger")

        with caplog.at_level(logging.WARNING):
            result = retry_function(
                mock_func, max_retries=3, retry_delay=0.1, error_prefix="Test operation failed", logger=logger
            )

        assert result == "success"
        assert "Test operation failed: fail 1" in caplog.text  # Initial attempt
        assert "Test operation failed (retry 1/3): fail 2" in caplog.text  # First retry
