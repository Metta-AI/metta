"""Tests for metta.common.util.tracing module."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from metta.common.util.tracing import Tracer, save_trace, trace, trace_events, tracer


class TestTraceDecorator:
    """Test cases for the @trace decorator."""

    def setup_method(self):
        """Clear trace events before each test."""
        global trace_events
        trace_events.clear()

    def test_trace_simple_function(self):
        """Test tracing a simple function."""
        @trace
        def add_numbers(a, b):
            return a + b
        
        result = add_numbers(3, 4)
        
        assert result == 7
        assert len(trace_events) == 2  # Begin and End events
        
        # Check begin event
        begin_event = trace_events[0]
        # Function name might be prefixed with test class in some contexts
        assert begin_event["name"].endswith("add_numbers")
        assert begin_event["category"] == "function"
        assert begin_event["ph"] == "B"
        assert "pid" in begin_event
        assert "tid" in begin_event
        assert "ts" in begin_event
        assert begin_event["args"]["filename"].endswith("test_tracing.py")
        assert begin_event["args"]["lineno"] > 0
        assert "stack_trace" in begin_event["args"]
        
        # Check end event
        end_event = trace_events[1]
        assert end_event["name"].endswith("add_numbers")
        assert end_event["category"] == "function"
        assert end_event["ph"] == "E"
        assert end_event["pid"] == begin_event["pid"]
        assert end_event["tid"] == begin_event["tid"]
        assert end_event["ts"] >= begin_event["ts"]

    def test_trace_method_with_class(self):
        """Test tracing a method bound to a class."""
        class Calculator:
            @trace
            def multiply(self, a, b):
                return a * b
        
        calc = Calculator()
        result = calc.multiply(5, 6)
        
        assert result == 30
        assert len(trace_events) == 2
        
        # Method should be named with class prefix
        begin_event = trace_events[0]
        assert begin_event["name"] == "Calculator.multiply"

    def test_trace_function_with_exception(self):
        """Test that tracing works even when function raises exception."""
        @trace
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Should still have begin event, but no end event due to exception
        assert len(trace_events) == 1
        assert trace_events[0]["ph"] == "B"
        assert trace_events[0]["name"] == "failing_function"

    def test_trace_nested_function_calls(self):
        """Test tracing nested function calls."""
        @trace
        def outer_function():
            return inner_function() + 1
        
        @trace
        def inner_function():
            return 42
        
        result = outer_function()
        
        assert result == 43
        assert len(trace_events) == 4  # outer_begin, inner_begin, inner_end, outer_end
        
        # Check order of events
        assert trace_events[0]["name"] == "outer_function"
        assert trace_events[0]["ph"] == "B"
        assert trace_events[1]["name"] == "inner_function"
        assert trace_events[1]["ph"] == "B"
        assert trace_events[2]["name"] == "inner_function"
        assert trace_events[2]["ph"] == "E"
        assert trace_events[3]["name"] == "outer_function"
        assert trace_events[3]["ph"] == "E"

    def test_trace_with_args_and_kwargs(self):
        """Test tracing function with various arguments."""
        @trace
        def complex_function(a, b, c=None, *args, **kwargs):
            return {"a": a, "b": b, "c": c, "args": args, "kwargs": kwargs}
        
        result = complex_function(1, 2, extra="value")
        
        expected = {"a": 1, "b": 2, "c": None, "args": (), "kwargs": {"extra": "value"}}
        assert result == expected
        assert len(trace_events) == 2
        assert trace_events[0]["name"].endswith("complex_function")

    def test_trace_preserves_function_metadata(self):
        """Test that tracing preserves original function's metadata."""
        @trace
        def documented_function():
            """This is a test function."""
            return "test"
        
        # The wrapper should preserve access to original function
        result = documented_function()
        assert result == "test"

    def test_trace_stack_trace_generation(self):
        """Test that stack trace is properly generated."""
        @trace
        def test_function():
            return "result"
        
        test_function()
        
        begin_event = trace_events[0]
        stack_trace = begin_event["args"]["stack_trace"]
        
        # Stack trace should be a string with function names separated by '>'
        assert isinstance(stack_trace, str)
        assert "test_function" not in stack_trace  # trace_wrapper is filtered out
        # Should contain the test method name
        assert "test_trace_stack_trace_generation" in stack_trace

    def test_trace_timing_information(self):
        """Test that timing information is reasonable."""
        @trace
        def slow_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        begin_event = trace_events[0]
        end_event = trace_events[1]
        
        # End should be after begin
        assert end_event["ts"] > begin_event["ts"]
        
        # Duration should be at least 10ms (10,000 microseconds)
        duration_us = end_event["ts"] - begin_event["ts"]
        assert duration_us >= 8000  # Allow some variance

    def test_trace_thread_information(self):
        """Test that thread and process information is captured."""
        @trace
        def threaded_function():
            return threading.get_ident()
        
        result = threaded_function()
        
        assert result == threading.get_ident()
        
        begin_event = trace_events[0]
        assert begin_event["pid"] == os.getpid()
        assert begin_event["tid"] == threading.get_ident()


class TestTracerContextManager:
    """Test cases for the Tracer context manager."""

    def setup_method(self):
        """Clear trace events before each test."""
        global trace_events
        trace_events.clear()

    def test_tracer_basic_usage(self):
        """Test basic usage of tracer context manager."""
        with tracer("test_section"):
            time.sleep(0.01)
            result = 42
        
        assert len(trace_events) == 1
        
        event = trace_events[0]
        assert event["name"] == "test_section"
        assert event["category"] == "function"
        assert event["ph"] == "X"  # Complete event
        assert "pid" in event
        assert "tid" in event
        assert "ts" in event
        assert "dur" in event
        
        # Duration should be reasonable (at least 10ms)
        assert event["dur"] >= 8000  # microseconds

    def test_tracer_nested_contexts(self):
        """Test nested tracer contexts."""
        with tracer("outer_section"):
            time.sleep(0.005)
            with tracer("inner_section"):
                time.sleep(0.005)
                result = "nested"
        
        assert len(trace_events) == 2
        
        # Events should be in chronological order by start time
        outer_event = next(e for e in trace_events if e["name"] == "outer_section")
        inner_event = next(e for e in trace_events if e["name"] == "inner_section")
        
        # Inner should start after outer
        assert inner_event["ts"] >= outer_event["ts"]
        
        # Inner should end before outer ends
        inner_end = inner_event["ts"] + inner_event["dur"]
        outer_end = outer_event["ts"] + outer_event["dur"]
        assert inner_end <= outer_end

    def test_tracer_with_exception(self):
        """Test tracer behavior when exception occurs."""
        with pytest.raises(RuntimeError, match="Test exception"):
            with tracer("failing_section"):
                raise RuntimeError("Test exception")
        
        # Should still record the event
        assert len(trace_events) == 1
        event = trace_events[0]
        assert event["name"] == "failing_section"
        assert event["dur"] >= 0

    def test_tracer_factory_function(self):
        """Test the tracer factory function."""
        context_manager = tracer("test_name")
        assert isinstance(context_manager, Tracer)
        assert context_manager.name == "test_name"

    def test_tracer_timing_accuracy(self):
        """Test that tracer timing is reasonably accurate."""
        sleep_duration = 0.02  # 20ms
        
        with tracer("timing_test"):
            time.sleep(sleep_duration)
        
        event = trace_events[0]
        duration_ms = event["dur"] / 1000  # Convert to milliseconds
        
        # Should be close to expected duration (allow some variance)
        assert 15 <= duration_ms <= 40  # 15-40ms range

    def test_tracer_multiple_sequential(self):
        """Test multiple sequential tracer contexts."""
        names = ["section1", "section2", "section3"]
        
        for name in names:
            with tracer(name):
                time.sleep(0.001)
        
        assert len(trace_events) == 3
        
        # Check all sections were recorded
        recorded_names = [event["name"] for event in trace_events]
        assert recorded_names == names

    def test_tracer_attributes(self):
        """Test Tracer class attributes and methods."""
        tracer_instance = Tracer("test")
        assert tracer_instance.name == "test"
        assert tracer_instance.start == 0
        
        # Test __enter__ and __exit__ manually
        tracer_instance.__enter__()
        assert tracer_instance.start > 0
        
        time.sleep(0.001)
        tracer_instance.__exit__(None, None, None)
        
        assert len(trace_events) == 1
        assert trace_events[0]["name"] == "test"


class TestSaveTrace:
    """Test cases for the save_trace function."""

    def setup_method(self):
        """Clear trace events before each test."""
        global trace_events
        trace_events.clear()

    def test_save_trace_basic(self):
        """Test basic save_trace functionality."""
        # Add some trace events
        @trace
        def test_function():
            return "test"
        
        test_function()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            save_trace(temp_path)
            
            # Read back the file
            with open(temp_path, "r") as f:
                data = json.load(f)
            
            assert "traceEvents" in data
            assert "displayTimeUnit" in data
            assert data["displayTimeUnit"] == "ms"
            assert len(data["traceEvents"]) == 2  # Begin and End events
            
            # Check structure of events
            for event in data["traceEvents"]:
                assert "name" in event
                assert "category" in event
                assert "ph" in event
                assert "pid" in event
                assert "tid" in event
                assert "ts" in event
        
        finally:
            Path(temp_path).unlink()

    def test_save_trace_empty_events(self):
        """Test saving trace with no events."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            save_trace(temp_path)
            
            with open(temp_path, "r") as f:
                data = json.load(f)
            
            assert data["traceEvents"] == []
            assert data["displayTimeUnit"] == "ms"
        
        finally:
            Path(temp_path).unlink()

    def test_save_trace_mixed_events(self):
        """Test saving trace with mixed event types."""
        # Add function trace
        @trace
        def traced_func():
            pass
        
        traced_func()
        
        # Add context trace
        with tracer("context_section"):
            pass
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            save_trace(temp_path)
            
            with open(temp_path, "r") as f:
                data = json.load(f)
            
            # Should have 3 events: function begin, function end, context complete
            assert len(data["traceEvents"]) == 3
            
            # Check event types
            phases = [event["ph"] for event in data["traceEvents"]]
            assert "B" in phases  # Begin
            assert "E" in phases  # End
            assert "X" in phases  # Complete
        
        finally:
            Path(temp_path).unlink()

    def test_save_trace_file_creation(self):
        """Test that save_trace creates file in non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create path in subdirectory that doesn't exist
            trace_path = Path(temp_dir) / "traces" / "test_trace.json"
            
            @trace
            def test_function():
                return "test"
            
            test_function()
            
            # Ensure parent directory exists first
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            save_trace(str(trace_path))
            
            assert trace_path.exists()
            assert trace_path.is_file()
            
            # Verify content
            with open(trace_path, "r") as f:
                data = json.load(f)
            
            assert "traceEvents" in data


class TestTraceEventsGlobalState:
    """Test cases for global trace_events management."""

    def setup_method(self):
        """Clear trace events before each test."""
        global trace_events
        trace_events.clear()

    def test_trace_events_persistence(self):
        """Test that trace events persist across multiple operations."""
        @trace
        def func1():
            return 1
        
        @trace
        def func2():
            return 2
        
        func1()
        assert len(trace_events) == 2
        
        func2()
        assert len(trace_events) == 4
        
        with tracer("section"):
            pass
        
        assert len(trace_events) == 5

    def test_trace_events_modification(self):
        """Test that trace_events can be modified externally."""
        @trace
        def test_func():
            return "test"
        
        test_func()
        initial_count = len(trace_events)
        
        # Clear events
        trace_events.clear()
        assert len(trace_events) == 0
        
        # Add more events
        test_func()
        assert len(trace_events) == 2


class TestTracingIntegration:
    """Integration tests for the tracing module."""

    def setup_method(self):
        """Clear trace events before each test."""
        global trace_events
        trace_events.clear()

    def test_realistic_tracing_scenario(self):
        """Test a realistic scenario with mixed tracing."""
        class DataProcessor:
            @trace
            def __init__(self):
                self.data = []
            
            @trace
            def process_batch(self, items):
                with tracer("validation"):
                    self._validate(items)
                
                with tracer("processing"):
                    for item in items:
                        self._process_item(item)
                
                return len(self.data)
            
            @trace
            def _validate(self, items):
                if not items:
                    raise ValueError("Empty batch")
            
            @trace
            def _process_item(self, item):
                self.data.append(item * 2)
        
        processor = DataProcessor()
        result = processor.process_batch([1, 2, 3])
        
        assert result == 3
        assert processor.data == [2, 4, 6]
        
        # Count different types of events
        function_events = [e for e in trace_events if e.get("ph") in ["B", "E"]]
        context_events = [e for e in trace_events if e.get("ph") == "X"]
        
        # Should have function traces and context traces
        assert len(function_events) > 0
        assert len(context_events) == 2  # validation and processing
        
        # Verify context events
        context_names = {e["name"] for e in context_events}
        assert "validation" in context_names
        assert "processing" in context_names

    def test_chrome_tracing_format_compliance(self):
        """Test that generated events comply with Chrome tracing format."""
        @trace
        def sample_function():
            with tracer("inner_work"):
                time.sleep(0.001)
            return "done"
        
        sample_function()
        
        # Verify all events have required Chrome tracing fields
        for event in trace_events:
            # Required fields
            assert "name" in event
            assert "cat" in event or "category" in event
            assert "ph" in event
            assert "pid" in event
            assert "tid" in event
            assert "ts" in event
            
            # Phase-specific requirements
            if event["ph"] == "X":  # Complete events need duration
                assert "dur" in event
            
            # Check data types
            assert isinstance(event["name"], str)
            assert isinstance(event["pid"], int)
            assert isinstance(event["tid"], int)
            assert isinstance(event["ts"], int)
            
            if "dur" in event:
                assert isinstance(event["dur"], int)
                assert event["dur"] >= 0
