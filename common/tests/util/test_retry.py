"""Tests for the retry utilities with exponential backoff and jitter."""

import time
from dataclasses import dataclass

import pytest

from metta.common.util.retry import calculate_backoff_delay, retry_function, retry_on_exception


@dataclass
class CallTracker:
    """Helper to track function calls and control behavior."""

    calls: list[tuple[tuple, dict]]
    results: list
    current_call: int = 0

    def __call__(self, *args, **kwargs):
        """Track call and return/raise based on results list."""
        self.calls.append((args, kwargs))
        result = self.results[min(self.current_call, len(self.results) - 1)]
        self.current_call += 1

        if isinstance(result, Exception):
            raise result
        return result

    @property
    def call_count(self):
        return len(self.calls)


class TestRetryFunction:
    """Test the retry_function utility."""

    def test_success_on_first_try(self):
        """Test immediate success without retries."""

        def successful_func():
            return "success"

        result = retry_function(successful_func, initial_delay=0.01)
        assert result == "success"

    def test_success_after_retries(self):
        """Test success after some retries."""
        tracker = CallTracker([], [Exception("fail"), "success"])

        result = retry_function(tracker, initial_delay=0.01)
        assert result == "success"
        assert tracker.call_count == 2

    def test_all_retries_fail(self):
        """Test when all retries fail."""

        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            retry_function(always_fails, max_retries=2, initial_delay=0.01)

    def test_specific_exceptions_only(self):
        """Test that only specific exceptions trigger retries."""
        tracker = CallTracker([], [ValueError("retry me"), RuntimeError("don't retry")])

        with pytest.raises(RuntimeError, match="don't retry"):
            retry_function(tracker, exceptions=(ValueError,), initial_delay=0.01)

        # Should have tried twice: ValueError was retried, RuntimeError was not
        assert tracker.call_count == 2

    def test_exponential_backoff_timing(self, monkeypatch):
        """Test exponential backoff with actual timing."""
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda x: sleep_calls.append(x))

        # Disable jitter for predictable results
        def mock_uniform(a, b):
            return b

        monkeypatch.setattr("random.uniform", mock_uniform)

        tracker = CallTracker([], [Exception("fail")] * 3 + ["success"])

        retry_function(
            tracker,
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
        )

        # Verify exponential delays: 1, 2, 4
        assert sleep_calls == [1.0, 2.0, 4.0]
        assert tracker.call_count == 4

    def test_max_delay_is_respected(self, monkeypatch):
        """Test that delays don't exceed max_delay."""
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda x: sleep_calls.append(x))
        monkeypatch.setattr("random.uniform", lambda a, b: b)  # Disable jitter

        tracker = CallTracker([], [Exception("fail")] * 5 + ["success"])

        retry_function(
            tracker,
            max_retries=5,
            initial_delay=1.0,
            max_delay=5.0,
            backoff_factor=2.0,
        )

        # Delays should be: 1, 2, 4, 5, 5 (capped at 5)
        assert sleep_calls == [1.0, 2.0, 4.0, 5.0, 5.0]


class TestRetryDecorator:
    """Test the retry_on_exception decorator."""

    def test_decorator_basic_functionality(self):
        """Test decorator works and preserves metadata."""
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

    def test_decorator_with_arguments(self):
        """Test decorator passes arguments correctly."""

        @retry_on_exception(max_retries=1, initial_delay=0.01)
        def add(a, b, offset=0):
            if add.first_call:
                add.first_call = False
                raise ValueError("First call fails")
            return a + b + offset

        add.first_call = True

        assert add(2, 3) == 5
        assert add(2, 3, offset=10) == 15

    def test_decorator_specific_exceptions(self):
        """Test decorator with specific exception types."""

        @retry_on_exception(exceptions=(ValueError,), initial_delay=0.01)
        def selective_retry():
            if hasattr(selective_retry, "called"):
                raise RuntimeError("This won't be retried")
            selective_retry.called = True
            raise ValueError("This will be retried")

        with pytest.raises(RuntimeError):
            selective_retry()


class TestBackoffCalculation:
    """Test the backoff delay calculation."""

    def test_exponential_growth(self):
        """Test exponential growth without jitter."""
        # Test with jitter disabled
        assert calculate_backoff_delay(0, jitter=False) == 1.0
        assert calculate_backoff_delay(1, jitter=False) == 2.0
        assert calculate_backoff_delay(2, jitter=False) == 4.0
        assert calculate_backoff_delay(3, jitter=False) == 8.0

    def test_custom_parameters(self):
        """Test with custom initial delay and backoff factor."""
        assert calculate_backoff_delay(0, initial_delay=5.0, jitter=False) == 5.0
        assert calculate_backoff_delay(2, initial_delay=5.0, jitter=False) == 20.0

        assert calculate_backoff_delay(1, backoff_factor=3.0, jitter=False) == 3.0
        assert calculate_backoff_delay(2, backoff_factor=3.0, jitter=False) == 9.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        assert calculate_backoff_delay(10, max_delay=10.0, jitter=False) == 10.0
        assert calculate_backoff_delay(5, initial_delay=100.0, max_delay=50.0, jitter=False) == 50.0

    def test_jitter_adds_randomness(self):
        """Test that jitter produces random delays within expected range."""
        # For attempt=2: 1.0 * 2^2 = 4.0
        delays = [calculate_backoff_delay(2, jitter=True) for _ in range(20)]

        # All delays should be between 0 and 4.0
        assert all(0 <= d <= 4.0 for d in delays)

        # Should have different values (very unlikely to be all the same)
        assert len(set(delays)) > 1

        # Average should be around 2.0 (middle of the range)
        avg_delay = sum(delays) / len(delays)
        assert 1.0 < avg_delay < 3.0
