"""Tests for the retry utilities with exponential backoff and jitter."""

from unittest.mock import Mock, patch

import pytest

from metta.common.util.retry import calculate_backoff_delay, retry_function, retry_on_exception


class TestRetryFunction:
    """Test the retry_function utility."""

    def test_basic_retry_scenarios(self):
        """Test basic retry scenarios: success, retry then success, and all fail."""
        # Success on first try
        mock_func = Mock(return_value="success")
        assert retry_function(mock_func, initial_delay=0.01) == "success"
        assert mock_func.call_count == 1

        # Success after retries
        mock_func = Mock(side_effect=[Exception("fail"), "success"])
        assert retry_function(mock_func, initial_delay=0.01) == "success"
        assert mock_func.call_count == 2

        # All retries fail
        mock_func = Mock(side_effect=Exception("always fails"))
        with pytest.raises(Exception, match="always fails"):
            retry_function(mock_func, max_retries=2, initial_delay=0.01)
        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_specific_exceptions(self):
        """Test that only specific exceptions trigger retries."""
        mock_func = Mock(side_effect=[ValueError("retry"), RuntimeError("don't retry")])

        with pytest.raises(RuntimeError, match="don't retry"):
            retry_function(mock_func, exceptions=(ValueError,), initial_delay=0.01)

        assert mock_func.call_count == 2  # ValueError retried, RuntimeError not

    @patch("time.sleep")
    @patch("random.uniform", side_effect=lambda a, b: b)  # Disable jitter
    def test_exponential_backoff(self, mock_random, mock_sleep):
        """Test exponential backoff timing."""
        mock_func = Mock(side_effect=[Exception("fail")] * 3 + ["success"])

        retry_function(
            mock_func,
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
        )

        # Verify exponential delays: 1, 2, 4
        sleep_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_delays == [1.0, 2.0, 4.0]


class TestRetryDecorator:
    """Test the retry_on_exception decorator."""

    def test_decorator_basic_functionality(self):
        """Test decorator works with functions and preserves metadata."""
        call_count = 0

        @retry_on_exception(max_retries=2, initial_delay=0.01)
        def flaky_function(value):
            """Test function docstring."""
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet!")
            return f"success: {value}"

        # Test it works
        assert flaky_function("test") == "success: test"
        assert call_count == 2

        # Test metadata preserved
        assert flaky_function.__name__ == "flaky_function"
        assert flaky_function.__doc__ == "Test function docstring."

    @patch("time.sleep")
    def test_decorator_uses_retry_function(self, mock_sleep):
        """Test that decorator properly uses retry_function internally."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])

        @retry_on_exception(initial_delay=0.5, max_retries=1)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once()  # Should sleep once between retries


class TestBackoffCalculation:
    """Test the backoff delay calculation."""

    def test_calculate_backoff_delay(self):
        """Test backoff calculation with and without jitter."""
        # Test exponential growth (with jitter disabled)
        with patch("random.uniform", side_effect=lambda a, b: b):
            assert calculate_backoff_delay(0) == 1.0
            assert calculate_backoff_delay(1) == 2.0
            assert calculate_backoff_delay(2) == 4.0
            assert calculate_backoff_delay(5, max_delay=10.0) == 10.0  # Capped at max

        # Test jitter adds randomness
        # For attempt=2 with initial_delay=4.0: 4.0 * 2^2 = 16.0
        delays = [calculate_backoff_delay(2, initial_delay=4.0) for _ in range(10)]
        assert all(0 <= d <= 16.0 for d in delays)  # Jitter between 0 and calculated delay
        assert len(set(delays)) > 1  # Should have different values
