"""Unit tests for tAXIOM I/O error handling."""

import time
from unittest.mock import Mock, patch

import pytest

from metta.sweep.axiom import Ctx, Pipeline


class TestIOErrorHandling:
    """Test suite for I/O operation error handling."""

    def test_io_timeout_raises_ioerror(self):
        """Test that I/O operations timeout and raise IOError."""
        def slow_io(input_data):
            # I/O operations MUST accept previous stage output
            time.sleep(5)
            return "data"
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .io("fetch_data", slow_io, timeout=0.1)
            .stage("process", lambda data: f"processed: {data}")
        )
        
        ctx = Ctx()
        with pytest.raises(IOError) as exc_info:
            pipeline.run(ctx)
        
        assert "timed out after 0.1 seconds" in str(exc_info.value)
        assert "fetch_data" in str(exc_info.value)

    def test_io_retry_on_failure(self):
        """Test that I/O operations retry on failure."""
        mock_func = Mock(side_effect=[
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            "success"
        ])
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .io("fetch_data", mock_func, num_retries=2)
            .stage("process", lambda data: f"processed: {data}")
        )
        
        ctx = Ctx()
        result = pipeline.run(ctx)
        
        assert result == "processed: success"
        assert mock_func.call_count == 3

    def test_io_fail_hard_after_retries(self):
        """Test that I/O operations fail hard after exhausting retries."""
        mock_func = Mock(side_effect=ValueError("Database connection failed"))
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .io("fetch_data", mock_func, num_retries=2)
            .stage("process", lambda data: f"processed: {data}")
        )
        
        ctx = Ctx()
        with pytest.raises(IOError) as exc_info:
            pipeline.run(ctx)
        
        assert "failed after 3 attempts" in str(exc_info.value)
        assert "Database connection failed" in str(exc_info.value)
        assert mock_func.call_count == 3

    def test_io_with_timeout_and_retry(self):
        """Test combined timeout and retry behavior."""
        call_count = 0
        
        def slow_then_fast(input_data):
            # I/O operations MUST accept previous stage output
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(2)  # Will timeout
            return "success"
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .io("fetch_data", slow_then_fast, timeout=0.1, num_retries=1)
            .stage("process", lambda data: f"processed: {data}")
        )
        
        ctx = Ctx()
        result = pipeline.run(ctx)
        
        assert result == "processed: success"
        assert call_count == 2

    def test_io_no_retry_when_num_retries_zero(self):
        """Test that I/O operations don't retry when num_retries=0."""
        mock_func = Mock(side_effect=ValueError("Error"))
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .io("fetch_data", mock_func, num_retries=0)
        )
        
        ctx = Ctx()
        with pytest.raises(IOError) as exc_info:
            pipeline.run(ctx)
        
        assert "failed after 1 attempts" in str(exc_info.value)
        assert mock_func.call_count == 1

    def test_normal_stages_no_retry(self):
        """Test that normal stages don't have retry behavior."""
        mock_func = Mock(side_effect=ValueError("Processing error"))
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"input": "test"})
            .stage("process", mock_func)
        )
        
        ctx = Ctx()
        with pytest.raises(ValueError) as exc_info:
            pipeline.run(ctx)
        
        assert str(exc_info.value) == "Processing error"
        assert mock_func.call_count == 1

    def test_io_exponential_backoff(self):
        """Test that retries use exponential backoff."""
        mock_func = Mock(side_effect=[
            ConnectionError("Error 1"),
            ConnectionError("Error 2"),
            "success"
        ])
        
        with patch('time.sleep') as mock_sleep:
            pipeline = (Pipeline()
                .stage("prepare", lambda: {"input": "test"})
                .io("fetch_data", mock_func, num_retries=2)
            )
            
            ctx = Ctx()
            result = pipeline.run(ctx)
            
            assert result == "success"
            # Check exponential backoff: 2^0=1, 2^1=2
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)

    def test_io_backoff_capped_at_10_seconds(self):
        """Test that exponential backoff is capped at 10 seconds."""
        mock_func = Mock(side_effect=[
            ConnectionError("Error 1"),
            ConnectionError("Error 2"),
            ConnectionError("Error 3"),
            ConnectionError("Error 4"),
            ConnectionError("Error 5"),
            "success"
        ])
        
        with patch('time.sleep') as mock_sleep:
            pipeline = (Pipeline()
                .io("fetch_data", mock_func, num_retries=5)
            )
            
            ctx = Ctx()
            result = pipeline.run(ctx)
            
            assert result == "success"
            # Check that later retries are capped at 10
            # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16->10
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [1, 2, 4, 8, 10]

    def test_io_stage_stores_config(self):
        """Test that I/O stages store timeout and num_retries config."""
        pipeline = Pipeline()
        pipeline.io("fetch", lambda: "data", timeout=5.0, num_retries=3)
        
        io_stage = pipeline.stages[0]
        assert io_stage.timeout == 5.0
        assert io_stage.num_retries == 3
        assert io_stage.stage_type == "io"

    def test_io_preserves_function_signature(self):
        """Test that I/O wrapper preserves the original function signature."""
        def my_io_func(x: int, y: str = "default") -> dict:
            """My IO function."""
            return {"x": x, "y": y}
        
        pipeline = Pipeline()
        pipeline.io("fetch", my_io_func, timeout=1.0)
        
        io_stage = pipeline.stages[0]
        # The wrapper should preserve the original function's metadata
        assert io_stage.func.__name__ == "my_io_func"
        assert io_stage.func.__doc__ == "My IO function."

    def test_io_error_includes_context(self):
        """Test that I/O errors include helpful context."""
        def failing_io(input_data):
            # I/O operations MUST accept previous stage output
            raise FileNotFoundError("config.json not found")
        
        pipeline = (Pipeline()
            .stage("prepare", lambda: {"config_path": "/etc/app.conf"})
            .io("load_config", failing_io, timeout=2.0, num_retries=1)
        )
        
        ctx = Ctx()
        with pytest.raises(IOError) as exc_info:
            pipeline.run(ctx)
        
        error_msg = str(exc_info.value)
        assert "load_config" in error_msg
        assert "failed after 2 attempts" in error_msg
        assert "timeout: 2.0s" in error_msg
        assert "config.json not found" in error_msg

    def test_io_respects_data_flow_contract(self):
        """Test that I/O operations respect node->node data flow contract."""
        def load_file(config_data):
            # I/O must receive and can use previous stage output
            assert "path" in config_data
            assert config_data["path"] == "/tmp/data.json"
            return {"content": "file data", "source": config_data["path"]}
        
        pipeline = (Pipeline()
            .stage("config", lambda: {"path": "/tmp/data.json", "format": "json"})
            .io("load", load_file)
            .stage("validate", lambda data: {**data, "valid": True})
        )
        
        ctx = Ctx()
        result = pipeline.run(ctx)
        
        assert result["content"] == "file data"
        assert result["source"] == "/tmp/data.json"
        assert result["valid"] is True

    @patch('builtins.print')
    def test_io_logs_retry_attempts(self, mock_print):
        """Test that retry attempts are logged."""
        mock_func = Mock(side_effect=[
            ConnectionError("Network error"),
            "success"
        ])
        
        pipeline = (Pipeline()
            .io("fetch_data", mock_func, num_retries=1)
        )
        
        ctx = Ctx()
        pipeline.run(ctx)
        
        # Check that retry was logged
        mock_print.assert_called()
        call_args = str(mock_print.call_args_list)
        assert "attempt 1/2" in call_args.lower()
        assert "Network error" in call_args