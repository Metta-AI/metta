import logging
import time

import pytest

from mettagrid.util.stopwatch import Stopwatch


@pytest.fixture
def stopwatch():
    """Stopwatch fixture with default logger."""
    return Stopwatch()


class TestStopwatch:
    """Test suite for Stopwatch class."""

    def test_initialization(self):
        """Test stopwatch initialization."""
        # Test with default logger
        sw = Stopwatch()
        assert isinstance(sw.logger, logging.Logger)
        assert sw.logger.name == "Stopwatch"
        assert "__global__" in sw._timers

        # Test with custom logger
        custom_logger = logging.getLogger("custom")
        sw2 = Stopwatch(logger=custom_logger)
        assert sw2.logger == custom_logger
        assert sw2.logger.name == "custom"

    def test_basic_timing(self, stopwatch):
        """Test basic start/stop timing."""
        # Start timing
        stopwatch.start()
        time.sleep(0.1)
        elapsed = stopwatch.stop()

        # Check elapsed time is reasonable
        assert 0.09 < elapsed < 0.2

        # Check total elapsed
        total = stopwatch.get_elapsed()
        assert total == pytest.approx(elapsed, abs=0.001)

    def test_named_timers(self, stopwatch):
        """Test using multiple named timers."""
        # Start multiple timers
        stopwatch.start("timer1")
        time.sleep(0.05)
        stopwatch.start("timer2")
        time.sleep(0.05)

        # Stop timer1
        elapsed1 = stopwatch.stop("timer1")
        assert 0.09 < elapsed1 < 0.15

        # Timer2 still running
        time.sleep(0.05)
        elapsed2 = stopwatch.stop("timer2")
        assert 0.09 < elapsed2 < 0.15

        # Check individual elapsed times
        assert stopwatch.get_elapsed("timer1") == pytest.approx(elapsed1, abs=0.001)
        assert stopwatch.get_elapsed("timer2") == pytest.approx(elapsed2, abs=0.001)

    def test_context_manager(self, stopwatch, caplog):
        """Test using stopwatch as context manager."""
        # Test with time() method
        with stopwatch.time("test_context"):
            time.sleep(0.1)

        elapsed = stopwatch.get_elapsed("test_context")
        assert 0.09 < elapsed < 0.2

        # Test with callable syntax and logging
        with caplog.at_level(logging.INFO):
            with stopwatch("test_callable", log=logging.INFO):
                time.sleep(0.1)

            # Check log output
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "INFO"
            assert "test_callable took" in caplog.records[0].message
            assert "s" in caplog.records[0].message  # Should show seconds

    def test_checkpoint_functionality(self, stopwatch):
        """Test checkpoint recording."""
        stopwatch.start("test_timer")
        time.sleep(0.1)

        # Record named checkpoint
        stopwatch.checkpoint(100, "checkpoint1", "test_timer")
        time.sleep(0.1)

        # Record anonymous checkpoint
        stopwatch.checkpoint(200, timer_name="test_timer")

        stopwatch.stop("test_timer")

        # Check checkpoints were recorded
        timer = stopwatch._get_timer("test_timer")
        assert "checkpoint1" in timer["checkpoints"]
        assert timer["checkpoints"]["checkpoint1"][1] == 100
        assert len(timer["checkpoints"]) == 2

        # Verify anonymous checkpoint naming
        assert any(k.startswith("_lap_") for k in timer["checkpoints"])

    def test_lap_functionality(self, stopwatch):
        """Test lap timing."""
        stopwatch.start("lap_timer")
        time.sleep(0.1)

        # First lap
        lap1_time = stopwatch.lap(100, "lap_timer")
        assert lap1_time > 0.09

        time.sleep(0.1)

        # Second lap
        lap2_time = stopwatch.lap(200, "lap_timer")
        assert 0.09 < lap2_time < 0.15  # Should be time since last lap

        stopwatch.stop("lap_timer")

    def test_rate_calculations(self, stopwatch):
        """Test rate calculation methods."""
        stopwatch.start("rate_timer")
        time.sleep(0.1)

        # Test basic rate
        rate = stopwatch.get_rate(100, "rate_timer")
        assert 800 < rate < 1200  # ~1000 steps/sec

        # Add checkpoint and test lap rate
        stopwatch.checkpoint(100, "checkpoint1", "rate_timer")
        time.sleep(0.1)

        lap_rate = stopwatch.get_lap_rate(200, "rate_timer")
        assert 800 < lap_rate < 1200  # ~1000 steps/sec for the lap

        stopwatch.stop("rate_timer")

    def test_reset_functionality(self, stopwatch):
        """Test resetting timers."""
        # Start and stop a timer
        stopwatch.start("reset_test")
        time.sleep(0.1)
        stopwatch.stop("reset_test")

        # Check it has elapsed time
        assert stopwatch.get_elapsed("reset_test") > 0

        # Reset specific timer
        stopwatch.reset("reset_test")
        assert stopwatch.get_elapsed("reset_test") == 0

        # Test reset_all
        stopwatch.start("timer1")
        stopwatch.start("timer2")
        time.sleep(0.1)
        stopwatch.stop("timer1")
        stopwatch.stop("timer2")

        stopwatch.reset_all()
        assert len(stopwatch._timers) == 1  # Only global timer
        assert "__global__" in stopwatch._timers

    def test_get_last_elapsed(self, stopwatch):
        """Test getting last elapsed time."""
        # First run
        stopwatch.start("last_test")
        time.sleep(0.1)
        elapsed1 = stopwatch.stop("last_test")

        # Second run
        stopwatch.start("last_test")
        time.sleep(0.05)
        elapsed2 = stopwatch.stop("last_test")

        # Check last elapsed
        last = stopwatch.get_last_elapsed("last_test")
        assert last == pytest.approx(elapsed2, abs=0.001)

        # Check total is sum
        total = stopwatch.get_elapsed("last_test")
        assert total == pytest.approx(elapsed1 + elapsed2, abs=0.01)

    def test_running_timer_elapsed(self, stopwatch):
        """Test getting elapsed time while timer is running."""
        stopwatch.start("running_test")
        time.sleep(0.1)

        # Get elapsed while running
        elapsed = stopwatch.get_elapsed("running_test")
        assert elapsed > 0.09

        # Get last elapsed while running
        last = stopwatch.get_last_elapsed("running_test")
        assert last > 0.09

        time.sleep(0.05)

        # Should be greater now
        elapsed2 = stopwatch.get_elapsed("running_test")
        assert elapsed2 > elapsed

        stopwatch.stop("running_test")

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (45, "45 sec"),
            (90, "1.5 min"),
            (3000, "50.0 min"),
            (3600, "1.0 hours"),
            (7200, "2.0 hours"),
            (86400, "1.0 days"),
            (172800, "2.0 days"),
        ],
    )
    def test_format_time(self, stopwatch, seconds, expected):
        """Test time formatting."""
        assert stopwatch.format_time(seconds) == expected

    def test_estimate_remaining(self, stopwatch):
        """Test remaining time estimation."""
        stopwatch.start("estimate_test")
        time.sleep(0.1)

        # Estimate at 25% completion
        remaining_seconds, remaining_str = stopwatch.estimate_remaining(25, 100, "estimate_test")

        # Should be about 0.3 seconds (0.1 elapsed for 25 steps, so 0.3 for remaining 75)
        assert 0.25 < remaining_seconds < 0.35
        assert "sec" in remaining_str

        stopwatch.stop("estimate_test")

    def test_log_progress(self, stopwatch, caplog):
        """Test progress logging."""
        stopwatch.start("progress_test")
        time.sleep(0.1)

        with caplog.at_level(logging.INFO):
            # Log progress
            stopwatch.log_progress(50, 100, "progress_test", "Test Progress")

            # Check log output
            assert len(caplog.records) == 1
            record = caplog.records[0]
            assert record.levelname == "INFO"
            assert "Test Progress" in record.message
            assert "[progress_test]" in record.message
            assert "50/100" in record.message
            assert "50.00%" in record.message
            assert "remaining" in record.message

        stopwatch.stop("progress_test")

    def test_summaries(self, stopwatch):
        """Test getting timer summaries."""
        # Create some timers with activity
        stopwatch.start("timer1")
        time.sleep(0.05)
        stopwatch.checkpoint(100, "check1", "timer1")
        stopwatch.stop("timer1")

        stopwatch.start("timer2")
        time.sleep(0.05)

        # Get individual summary
        summary1 = stopwatch.get_summary("timer1")
        assert summary1["name"] == "timer1"
        assert not summary1["is_running"]
        assert "check1" in summary1["checkpoints"]

        # Get all summaries
        all_summaries = stopwatch.get_all_summaries()
        assert "timer1" in all_summaries
        assert "timer2" in all_summaries
        assert all_summaries["timer2"]["is_running"]

        stopwatch.stop("timer2")

    def test_get_all_elapsed(self, stopwatch):
        """Test getting all elapsed times."""
        # Create timers
        stopwatch.start("timer1")
        time.sleep(0.05)
        stopwatch.stop("timer1")

        stopwatch.start("timer2")
        time.sleep(0.05)
        stopwatch.stop("timer2")

        # Get all elapsed
        all_elapsed = stopwatch.get_all_elapsed(exclude_global=True)
        assert "timer1" in all_elapsed
        assert "timer2" in all_elapsed
        assert "__global__" not in all_elapsed

        # Include global
        all_elapsed_with_global = stopwatch.get_all_elapsed(exclude_global=False)
        assert "__global__" in all_elapsed_with_global

    def test_edge_cases(self, stopwatch, caplog):
        """Test edge cases and error handling."""
        with caplog.at_level(logging.WARNING):
            # Stop timer that's not running
            elapsed = stopwatch.stop("nonexistent")
            assert elapsed == 0.0
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "WARNING"
            assert "Timer 'nonexistent' not running" in caplog.records[0].message

            # Clear for next test
            caplog.clear()

            # Start timer that's already running
            stopwatch.start("double_start")
            stopwatch.start("double_start")
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "WARNING"
            assert "Timer 'double_start' already running" in caplog.records[0].message

            # Clear for next test
            caplog.clear()

            # Checkpoint on non-running timer
            stopwatch.checkpoint(100, "check1", "not_running")
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "WARNING"
            assert "Timer 'not_running' not running" in caplog.records[0].message

        # Rate with zero elapsed time
        rate = stopwatch.get_rate(100, "zero_timer")
        assert rate == 0.0

        # Cleanup
        stopwatch.stop("double_start")

    def test_global_timer(self, stopwatch):
        """Test global timer behavior."""
        # Test with None (should use global)
        stopwatch.start()
        time.sleep(0.1)
        elapsed = stopwatch.stop()

        assert elapsed > 0.09
        assert stopwatch.get_elapsed() == elapsed

        # Reset global timer
        stopwatch.reset()
        assert stopwatch.get_elapsed() == 0.0


class TestStopwatchIntegration:
    """Integration tests for more complex scenarios."""

    def test_multiple_concurrent_timers(self):
        """Test managing multiple concurrent timers."""
        sw = Stopwatch()

        # Start multiple timers in sequence
        sw.start("download")
        time.sleep(0.05)

        sw.start("processing")
        time.sleep(0.05)

        sw.start("upload")
        time.sleep(0.05)

        # Stop in different order
        sw.stop("processing")
        sw.stop("download")
        sw.stop("upload")

        # Verify all have different elapsed times
        elapsed = sw.get_all_elapsed()
        assert elapsed["download"] > elapsed["processing"]
        assert elapsed["processing"] > elapsed["upload"]

    def test_lap_rate_tracking(self):
        """Test tracking rates across multiple laps."""
        sw = Stopwatch()
        sw.start("training")

        # First lap: 100 steps in 0.1 seconds
        time.sleep(0.1)
        lap1_time = sw.lap(100, "training")
        assert 0.09 < lap1_time < 0.11

        # Second lap: 200 more steps (total 300) in another 0.1 seconds
        time.sleep(0.1)
        lap2_time = sw.lap(300, "training")
        assert 0.09 < lap2_time < 0.11

        # Third lap: 300 more steps (total 600) in another 0.1 seconds
        time.sleep(0.1)
        lap3_time = sw.lap(600, "training")
        assert 0.09 < lap3_time < 0.11

        # Now calculate rates BETWEEN checkpoints
        # Move forward a bit in time so we can calculate rates
        time.sleep(0.05)

        # Get current lap rate (should be based on steps since last checkpoint)
        current_rate = sw.get_lap_rate(650, "training")  # 50 steps in ~0.05 seconds
        assert 800 < current_rate < 1200  # ~1000 steps/sec

        sw.stop("training")

        # Verify the checkpoint data
        timer = sw._get_timer("training")
        checkpoints = timer["checkpoints"]
        assert len(checkpoints) == 3

        # Extract checkpoint data for verification
        checkpoint_list = sorted(checkpoints.items(), key=lambda x: x[1][0])
        assert checkpoint_list[0][1][1] == 100  # First checkpoint at 100 steps
        assert checkpoint_list[1][1][1] == 300  # Second checkpoint at 300 steps
        assert checkpoint_list[2][1][1] == 600  # Third checkpoint at 600 steps

    def test_real_world_scenario(self):
        """Test a realistic usage scenario."""
        sw = Stopwatch()

        # Simulate a data processing pipeline
        sw.start()  # Start global timer
        sw.start("load_data")
        time.sleep(0.05)
        sw.stop("load_data")

        sw.start("process_data")
        for i in range(3):
            sw.checkpoint(i * 100, f"batch_{i}", "process_data")
            time.sleep(0.03)
        sw.stop("process_data")

        sw.start("save_results")
        time.sleep(0.02)
        sw.stop("save_results")
        sw.stop()  # Stop global timer

        # Verify timing relationships
        all_elapsed = sw.get_all_elapsed(exclude_global=False)  # Include global timer
        total_time = all_elapsed["__global__"]  # total timer
        component_sum = all_elapsed["load_data"] + all_elapsed["process_data"] + all_elapsed["save_results"]

        # Total should be approximately the sum of components
        assert total_time == pytest.approx(component_sum, rel=0.1)

        # Process should be longest
        assert all_elapsed["process_data"] > all_elapsed["load_data"]
        assert all_elapsed["process_data"] > all_elapsed["save_results"]

    def test_logging_scenarios(self, caplog):
        """Test various logging scenarios in integration."""
        sw = Stopwatch()

        with caplog.at_level(logging.INFO):
            # Test multiple operations with logging
            with sw("operation1", log=logging.INFO):
                time.sleep(0.05)

            with sw("operation2", log=logging.INFO):
                time.sleep(0.03)

            # Test progress logging
            sw.start("batch_process")
            for i in range(3):
                time.sleep(0.02)
                sw.log_progress((i + 1) * 33, 100, "batch_process", "Batch Processing")
            sw.stop("batch_process")

            # Verify all expected logs
            info_records = [r for r in caplog.records if r.levelname == "INFO"]
            assert len(info_records) == 5  # 2 context managers + 3 progress logs

            # Check operation logs
            operation_logs = [r.message for r in info_records if "took" in r.message]
            assert len(operation_logs) == 2
            assert any("operation1" in msg for msg in operation_logs)
            assert any("operation2" in msg for msg in operation_logs)

            # Check progress logs
            progress_logs = [r.message for r in info_records if "Batch Processing" in r.message]
            assert len(progress_logs) == 3
            assert any("33/100" in msg for msg in progress_logs)
            assert any("66/100" in msg for msg in progress_logs)
            assert any("99/100" in msg for msg in progress_logs)


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean state between tests."""
    yield
    # Cleanup after each test if needed
