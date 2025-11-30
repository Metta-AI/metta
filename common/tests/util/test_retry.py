"""Tests for the retry utilities using tenacity with exponential backoff."""

from dataclasses import dataclass

import pytest

from metta.common.util.retry import retry_function, retry_on_exception


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

    def test_max_retries_respected(self):
        """Test that max_retries limits the number of attempts."""
        tracker = CallTracker([], [Exception("fail")] * 10)

        with pytest.raises(Exception, match="fail"):
            retry_function(tracker, max_retries=3, initial_delay=0.01)

        # Should have tried initial + 3 retries = 4 times
        assert tracker.call_count == 4


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
