"""
Tests for retry utilities.
"""

import time
import pytest
from unittest.mock import Mock, patch
import logging

from metta.utils.retry import exponential_backoff_retry, env_init_retry


class TestExponentialBackoffRetry:
    """Test the exponential backoff retry decorator."""
    
    def test_successful_first_attempt(self):
        """Test function that succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        @exponential_backoff_retry(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """Test function that fails once then succeeds."""
        mock_func = Mock(side_effect=[RuntimeError("fail"), "success"])
        
        @exponential_backoff_retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with patch('time.sleep') as mock_sleep:
            result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once_with(0.01)
    
    def test_all_attempts_fail(self):
        """Test function that fails all attempts."""
        mock_func = Mock(side_effect=RuntimeError("always fails"))
        
        @exponential_backoff_retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with patch('time.sleep'):
            with pytest.raises(RuntimeError, match="always fails"):
                test_func()
        
        assert mock_func.call_count == 3
    
    def test_exponential_backoff_timing(self):
        """Test that delays increase exponentially."""
        mock_func = Mock(side_effect=[
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            "success"
        ])
        
        @exponential_backoff_retry(
            max_attempts=3,
            initial_delay=0.1,
            backoff_factor=2.0
        )
        def test_func():
            return mock_func()
        
        sleep_calls = []
        
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        with patch('time.sleep', side_effect=mock_sleep):
            result = test_func()
        
        assert result == "success"
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 0.1  # First retry delay
        assert sleep_calls[1] == 0.2  # Second retry delay (0.1 * 2)
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        mock_func = Mock(side_effect=[
            RuntimeError("fail") for _ in range(5)
        ] + ["success"])
        
        @exponential_backoff_retry(
            max_attempts=6,
            initial_delay=0.5,
            max_delay=1.0,
            backoff_factor=3.0
        )
        def test_func():
            return mock_func()
        
        sleep_calls = []
        
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        with patch('time.sleep', side_effect=mock_sleep):
            result = test_func()
        
        assert result == "success"
        # Check that delays don't exceed max_delay
        assert sleep_calls[0] == 0.5   # First retry
        assert sleep_calls[1] == 1.0   # Second retry (capped)
        assert all(delay <= 1.0 for delay in sleep_calls)
    
    def test_specific_exception_handling(self):
        """Test that only specified exceptions trigger retry."""
        mock_func = Mock(side_effect=[
            ConnectionError("connection failed"),
            ValueError("wrong value"),  # This should not be retried
        ])
        
        @exponential_backoff_retry(
            max_attempts=3,
            initial_delay=0.01,
            exceptions=(ConnectionError, TimeoutError)
        )
        def test_func():
            return mock_func()
        
        with patch('time.sleep'):
            with pytest.raises(ValueError, match="wrong value"):
                test_func()
        
        # Should only be called twice: once for ConnectionError, once for ValueError
        assert mock_func.call_count == 2
    
    def test_function_with_arguments(self):
        """Test retry decorator with function arguments."""
        mock_func = Mock(side_effect=[RuntimeError("fail"), "success"])
        
        @exponential_backoff_retry(max_attempts=2, initial_delay=0.01)
        def test_func(arg1, arg2, kwarg1=None):
            return mock_func(arg1, arg2, kwarg1=kwarg1)
        
        with patch('time.sleep'):
            result = test_func("a", "b", kwarg1="c")
        
        assert result == "success"
        assert mock_func.call_count == 2
        # Check that arguments were passed correctly both times
        for call in mock_func.call_args_list:
            assert call[0] == ("a", "b")
            assert call[1] == {"kwarg1": "c"}
    
    def test_logging(self, caplog):
        """Test that retry attempts are logged."""
        mock_func = Mock(side_effect=[
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            "success"
        ])
        
        @exponential_backoff_retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with patch('time.sleep'):
            with caplog.at_level(logging.WARNING):
                result = test_func()
        
        assert result == "success"
        # Check that warnings were logged
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 2
        assert "Attempt 1/3 failed" in caplog.text
        assert "Attempt 2/3 failed" in caplog.text
        assert "Retrying in" in caplog.text


class TestEnvInitRetry:
    """Test the convenience env_init_retry decorator."""
    
    def test_env_init_retry_defaults(self):
        """Test env_init_retry with default settings."""
        mock_func = Mock(side_effect=[
            ConnectionError("connection lost"),
            RuntimeError("init failed"),
            "success"
        ])
        
        @env_init_retry
        def init_environment():
            return mock_func()
        
        with patch('time.sleep'):
            result = init_environment()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_env_init_retry_non_retriable_exception(self):
        """Test that env_init_retry doesn't retry on non-specified exceptions."""
        mock_func = Mock(side_effect=ValueError("invalid config"))
        
        @env_init_retry
        def init_environment():
            return mock_func()
        
        with pytest.raises(ValueError, match="invalid config"):
            init_environment()
        
        # Should only be called once since ValueError is not in the retry list
        assert mock_func.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
