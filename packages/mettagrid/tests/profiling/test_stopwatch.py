import logging
import time

import pytest

from mettagrid.profiling.stopwatch import Checkpoint, Stopwatch, with_instance_timer, with_timer


@pytest.fixture
def stopwatch():
    """Stopwatch fixture with logging enabled."""
    return Stopwatch(log_level=logging.INFO)


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean state between tests."""
    yield
    # Cleanup after each test if needed


class TestStopwatch:
    """Test suite for Stopwatch class."""

    def test_basic_timing(self, stopwatch: Stopwatch):
        """Test basic start/stop timing."""
        # Start timing
        start_time = time.time()
        stopwatch.start()
        time.sleep(0.1)
        end_time = time.time()
        elapsed = stopwatch.stop()
        actual_elapsed = end_time - start_time

        # Check elapsed time matches actual measured time
        assert elapsed == pytest.approx(actual_elapsed, abs=0.01)
        # Sleep can only take longer, never shorter
        assert elapsed >= 0.1

        # Check total elapsed
        total = stopwatch.get_elapsed()
        assert total == pytest.approx(elapsed, abs=0.001)

    def test_named_timers(self, stopwatch: Stopwatch):
        """Test using multiple named timers."""
        # Start multiple timers
        stopwatch.start("timer1")
        time.sleep(0.05)
        stopwatch.start("timer2")
        time.sleep(0.05)

        # Stop timer1
        elapsed1 = stopwatch.stop("timer1")
        assert elapsed1 == pytest.approx(0.1, abs=0.1)

        # Timer2 still running
        time.sleep(0.05)
        elapsed2 = stopwatch.stop("timer2")
        assert elapsed2 == pytest.approx(0.1, abs=0.1)

        # Check individual elapsed times
        assert stopwatch.get_elapsed("timer1") == pytest.approx(elapsed1, abs=0.1)
        assert stopwatch.get_elapsed("timer2") == pytest.approx(elapsed2, abs=0.1)

    def test_context_manager(self, caplog: pytest.LogCaptureFixture):
        """Test using stopwatch as context manager."""
        stopwatch = Stopwatch(log_level=logging.INFO)

        # Test with time() method
        with stopwatch.time("test_context"):
            time.sleep(0.1)

        elapsed = stopwatch.get_elapsed("test_context")
        assert elapsed == pytest.approx(0.1, abs=0.1)

        # Test with callable syntax and logging
        with caplog.at_level(logging.INFO, logger=stopwatch.logger.name):
            caplog.clear()

            with stopwatch("test_callable", log_level=logging.INFO):
                time.sleep(0.1)

            # Check log output
            stopwatch_logs = [r for r in caplog.records if r.name == stopwatch.logger.name]
            assert len(stopwatch_logs) == 1
            assert stopwatch_logs[0].levelname == "INFO"
            assert "test_callable took" in stopwatch_logs[0].message
            assert "s" in stopwatch_logs[0].message  # Should show seconds

    def test_checkpoint_functionality(self, stopwatch: Stopwatch):
        """Test checkpoint recording."""
        stopwatch.start("test_timer")
        time.sleep(0.1)

        # Record named checkpoint
        stopwatch.checkpoint(100, "checkpoint1", "test_timer")
        time.sleep(0.1)

        # Record anonymous checkpoint
        stopwatch.checkpoint(200, name="test_timer")

        stopwatch.stop("test_timer")

        # Check checkpoints were recorded
        timer = stopwatch._get_timer("test_timer")
        assert "checkpoint1" in timer.checkpoints
        assert timer.checkpoints["checkpoint1"]["steps"] == 100
        assert len(timer.checkpoints) == 3  # with _start

        # Verify anonymous checkpoint naming
        assert any(k.startswith("_lap_") for k in timer.checkpoints)

    def test_lap_functionality(self, stopwatch: Stopwatch):
        """Test lap functionality."""
        stopwatch.start("lap_timer")

        # First lap
        lap1_start = time.time()
        time.sleep(0.1)
        lap1_end = time.time()
        lap1_time = stopwatch.lap(100, "lap_timer")
        lap1_actual = lap1_end - lap1_start

        assert lap1_time == pytest.approx(lap1_actual, abs=0.01)
        assert lap1_time >= 0.1

        # Second lap
        lap2_start = time.time()
        time.sleep(0.1)
        lap2_end = time.time()
        lap2_time = stopwatch.lap(200, "lap_timer")
        lap2_actual = lap2_end - lap2_start

        assert lap2_time == pytest.approx(lap2_actual, abs=0.01)
        assert lap2_time >= 0.1

        stopwatch.stop("lap_timer")

    def test_rate_calculations(self, stopwatch: Stopwatch):
        """Test rate calculation methods."""
        # Start timing
        start_time = time.time()
        stopwatch.start("rate_timer")
        time.sleep(0.1)
        mid_time = time.time()
        actual_elapsed = mid_time - start_time

        # Test basic rate
        steps = 100
        rate = stopwatch.get_rate(steps, "rate_timer")
        expected_rate = steps / actual_elapsed
        assert rate == pytest.approx(expected_rate, rel=0.01)

        # Add checkpoint and test lap rate
        stopwatch.checkpoint(steps, "checkpoint1", "rate_timer")

        checkpoint_time = time.time()
        time.sleep(0.1)
        lap_end_time = time.time()
        lap_elapsed = lap_end_time - checkpoint_time

        # Test lap rate
        additional_steps = 100
        total_steps = steps + additional_steps
        lap_rate = stopwatch.get_lap_rate(total_steps, "rate_timer")
        expected_lap_rate = additional_steps / lap_elapsed
        assert lap_rate == pytest.approx(expected_lap_rate, rel=0.01)

        stopwatch.stop("rate_timer")

    def test_reset_functionality(self, stopwatch: Stopwatch):
        """Test resetting timers."""
        # Start and stop a timer
        stopwatch.start("A")
        time.sleep(0.1)
        stopwatch.stop("A")

        # Check it has elapsed time
        assert stopwatch.get_elapsed("A") > 0

        # Reset specific timer
        stopwatch.reset("A")
        assert stopwatch.get_elapsed("A") == 0

        # Test reset_all
        stopwatch.start("B")
        stopwatch.start("C")
        time.sleep(0.1)
        stopwatch.stop("B")
        stopwatch.stop("C")

        stopwatch.reset_all()
        assert len(stopwatch._timers) == 4  # 3 + global; reset zeroes all timers
        assert stopwatch.GLOBAL_TIMER_NAME in stopwatch._timers

    def test_get_last_elapsed(self, stopwatch: Stopwatch):
        """Test getting last elapsed time."""
        # First run - measure actual time
        start1 = time.time()
        stopwatch.start("last_test")
        time.sleep(0.1)
        end1 = time.time()
        elapsed1 = stopwatch.stop("last_test")
        actual_elapsed1 = end1 - start1

        # Verify first run elapsed time matches actual
        assert elapsed1 == pytest.approx(actual_elapsed1, abs=0.01)

        # Second run - measure actual time
        start2 = time.time()
        stopwatch.start("last_test")
        time.sleep(0.05)
        end2 = time.time()
        elapsed2 = stopwatch.stop("last_test")
        actual_elapsed2 = end2 - start2

        # Verify second run elapsed time matches actual
        assert elapsed2 == pytest.approx(actual_elapsed2, abs=0.01)

        # Check last elapsed matches the second run
        last = stopwatch.get_last_elapsed("last_test")
        assert last == pytest.approx(elapsed2, abs=0.01)
        assert last == pytest.approx(actual_elapsed2, abs=0.01)

        # Check total is sum of both runs
        total = stopwatch.get_elapsed("last_test")
        expected_total = actual_elapsed1 + actual_elapsed2
        assert total == pytest.approx(expected_total, abs=0.02)
        assert total == pytest.approx(elapsed1 + elapsed2, abs=0.01)

    def test_running_timer_elapsed(self, stopwatch: Stopwatch):
        """Test getting elapsed time while timer is running."""
        stopwatch.start("running_test")
        time.sleep(0.1)

        # Get elapsed while running
        elapsed = stopwatch.get_elapsed("running_test")
        assert elapsed >= 0.1

        # Get last elapsed while running
        last = stopwatch.get_last_elapsed("running_test")
        assert last >= 0.1

        time.sleep(0.05)

        # Should be greater now
        elapsed2 = stopwatch.get_elapsed("running_test")
        assert elapsed2 > elapsed
        assert elapsed2 >= 0.15

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
    def test_format_time(self, stopwatch: Stopwatch, seconds: float, expected: str):
        """Test time formatting."""
        assert stopwatch.format_time(seconds) == expected

    def test_estimate_remaining(self, stopwatch: Stopwatch):
        """Test remaining time estimation."""
        steps_completed = 25
        total_steps = 100

        stopwatch.start("estimate_test")
        time.sleep(0.1)

        elapsed = stopwatch.get_elapsed("estimate_test")

        # If 25 steps took 'elapsed' seconds, remaining 75 steps will take:
        remaining_steps = total_steps - steps_completed
        expected_remaining = (elapsed / steps_completed) * remaining_steps

        remaining_seconds, remaining_str = stopwatch.estimate_remaining(steps_completed, total_steps, "estimate_test")

        assert remaining_seconds == pytest.approx(expected_remaining, rel=0.01)
        assert "sec" in remaining_str

        stopwatch.stop("estimate_test")

    def test_log_progress(self, caplog: pytest.LogCaptureFixture):
        """Test progress logging."""
        stopwatch = Stopwatch(log_level=logging.INFO)

        stopwatch.start("progress_test")
        time.sleep(0.1)

        with caplog.at_level(logging.INFO, logger=stopwatch.logger.name):
            caplog.clear()

            # Log progress
            stopwatch.log_progress(50, 100, "progress_test", "Test Progress")

            # Check log output
            stopwatch_logs = [r for r in caplog.records if r.name == stopwatch.logger.name]
            assert len(stopwatch_logs) == 1
            record = stopwatch_logs[0]
            assert record.levelname == "INFO"
            assert "Test Progress" in record.message
            assert "[progress_test]" in record.message
            assert "50/100" in record.message
            assert "50.00%" in record.message
            assert "remaining" in record.message

        stopwatch.stop("progress_test")

    def test_summaries(self, stopwatch: Stopwatch):
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

    def test_get_all_elapsed(self, stopwatch: Stopwatch):
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
        assert stopwatch.GLOBAL_TIMER_NAME not in all_elapsed

        # Include global
        all_elapsed_with_global = stopwatch.get_all_elapsed(exclude_global=False)
        assert stopwatch.GLOBAL_TIMER_NAME in all_elapsed_with_global

    def test_edge_cases(self, caplog: pytest.LogCaptureFixture):
        """Test edge cases and error handling."""
        stopwatch = Stopwatch(log_level=logging.INFO)

        with caplog.at_level(logging.WARNING, logger=stopwatch.logger.name):
            caplog.clear()

            # Stop timer that's not running
            elapsed = stopwatch.stop("nonexistent")
            assert elapsed == 0.0

            stopwatch_logs = [r for r in caplog.records if r.name == stopwatch.logger.name]
            assert len(stopwatch_logs) == 1
            record = stopwatch_logs[0]
            assert record.levelname == "WARNING"
            assert "Timer 'nonexistent' not running" in record.message

            # Clear for next test
            caplog.clear()

            # Start timer that's already running
            stopwatch.start("double_start")
            stopwatch.start("double_start")

            stopwatch_logs = [r for r in caplog.records if r.name == stopwatch.logger.name]
            assert len(stopwatch_logs) == 1
            record = stopwatch_logs[0]
            assert record.levelname == "WARNING"
            assert "Timer 'double_start' already running" in record.message

        # Rate with zero elapsed time
        rate = stopwatch.get_rate(100, "zero_timer")
        assert rate == 0.0

        # Cleanup
        stopwatch.stop("double_start")

    def test_global_timer(self, stopwatch: Stopwatch):
        """Test global timer behavior."""
        start_time = time.time()
        stopwatch.start()
        time.sleep(0.1)
        end_time = time.time()
        elapsed = stopwatch.stop()
        actual_elapsed = end_time - start_time

        assert elapsed == pytest.approx(actual_elapsed, abs=0.01)
        assert elapsed >= 0.1

        assert stopwatch.get_elapsed() == pytest.approx(elapsed, abs=0.001)

        # Reset global timer
        stopwatch.reset()
        assert stopwatch.get_elapsed() == 0.0

    def test_get_filename(self, stopwatch: Stopwatch):
        """Test get_filename method."""
        # Test unknown (no references)
        assert stopwatch.get_filename("unknown_timer") == "unknown"

        # Test single file
        stopwatch.start("single_file")
        stopwatch.stop("single_file")
        filename = stopwatch.get_filename("single_file")
        assert filename.endswith("test_stopwatch.py")

        # Test multifile
        # We simulate multiple references by manually adding them
        timer = stopwatch._get_timer("multifile_test")
        timer.references.add(("file1.py", 10))
        timer.references.add(("file2.py", 20))
        assert stopwatch.get_filename("multifile_test") == "multifile"

        # Test multiple references from same file
        timer2 = stopwatch._get_timer("samefile_test")
        timer2.references.add(("same.py", 10))
        timer2.references.add(("same.py", 20))
        assert stopwatch.get_filename("samefile_test") == "same.py"

    def test_lap_all_with_running_and_stopped_timers(self, stopwatch: Stopwatch):
        tol = 0.02  # Keep tight tolerance for comparing against actual measurements

        # Track when each timer was last started/stopped
        timer_states = {
            "A": {"running": False, "elapsed": 0.0, "last_start": None},
            "B": {"running": False, "elapsed": 0.0, "last_start": None},
            "C": {"running": False, "elapsed": 0.0, "last_start": None},
        }

        def update_timer_state(name, action, current_time):
            """Helper to track actual elapsed time for each timer"""
            state = timer_states[name]
            if action == "start" and not state["running"]:
                state["running"] = True
                state["last_start"] = current_time
            elif action == "stop" and state["running"]:
                state["elapsed"] += current_time - state["last_start"]
                state["running"] = False
                state["last_start"] = None

        def get_current_elapsed(name, current_time):
            """Get the actual elapsed time for a timer"""
            state = timer_states[name]
            elapsed = state["elapsed"]
            if state["running"]:
                elapsed += current_time - state["last_start"]
            return elapsed

        # Start all timers and track actual start time
        start_time = time.time()
        stopwatch.start("A")
        update_timer_state("A", "start", start_time)
        stopwatch.start("B")
        update_timer_state("B", "start", start_time)
        stopwatch.start("C")
        update_timer_state("C", "start", start_time)

        # Sleep 0.1s - all timers running
        time.sleep(0.1)
        checkpoint1_time = time.time()

        # Stop A
        stopwatch.stop("A")
        update_timer_state("A", "stop", checkpoint1_time)

        # Sleep 0.1s - B and C running
        time.sleep(0.1)
        checkpoint2_time = time.time()

        # Stop B, Start A
        stopwatch.stop("B")
        update_timer_state("B", "stop", checkpoint2_time)
        stopwatch.start("A")
        update_timer_state("A", "start", checkpoint2_time)

        # Sleep 0.1s - A and C running
        time.sleep(0.1)
        lap1_time = time.time()

        # First lap_all
        lap_times = stopwatch.lap_all(1000)

        # Verify lap times match actual elapsed times
        actual_A = get_current_elapsed("A", lap1_time)
        actual_B = get_current_elapsed("B", lap1_time)
        actual_C = get_current_elapsed("C", lap1_time)

        assert lap_times["A"] == pytest.approx(actual_A, abs=tol)
        assert lap_times["B"] == pytest.approx(actual_B, abs=tol)
        assert lap_times["C"] == pytest.approx(actual_C, abs=tol)

        # Sleep 0.1s - A and C running
        time.sleep(0.1)
        checkpoint3_time = time.time()

        # Stop A
        stopwatch.stop("A")
        update_timer_state("A", "stop", checkpoint3_time)

        # Sleep 0.1s - only C running
        time.sleep(0.1)
        lap2_time = time.time()

        # Store previous elapsed times for delta calculation
        prev_A = actual_A
        prev_B = actual_B
        prev_C = actual_C

        # Second lap_all
        lap_times_2 = stopwatch.lap_all(2000)

        # Calculate actual deltas since last lap
        actual_A2 = get_current_elapsed("A", lap2_time)
        actual_B2 = get_current_elapsed("B", lap2_time)
        actual_C2 = get_current_elapsed("C", lap2_time)

        delta_A = actual_A2 - prev_A
        delta_B = actual_B2 - prev_B
        delta_C = actual_C2 - prev_C

        assert lap_times_2["A"] == pytest.approx(delta_A, abs=tol)
        assert lap_times_2["B"] == pytest.approx(delta_B, abs=tol)
        assert lap_times_2["C"] == pytest.approx(delta_C, abs=tol)

        # Start A
        stopwatch.start("A")
        update_timer_state("A", "start", lap2_time)

        # Sleep 0.1s - A and C running
        time.sleep(0.1)
        checkpoint4_time = time.time()

        # Stop A and C
        stopwatch.stop("A")
        update_timer_state("A", "stop", checkpoint4_time)
        stopwatch.stop("C")
        update_timer_state("C", "stop", checkpoint4_time)

        # Sleep 0.1s - all stopped
        time.sleep(0.1)
        lap3_time = time.time()

        # Store previous elapsed times for delta calculation
        prev_A2 = actual_A2
        prev_B2 = actual_B2
        prev_C2 = actual_C2

        # Third lap_all
        lap_times_3 = stopwatch.lap_all(3000)

        # Calculate actual deltas since last lap
        actual_A3 = get_current_elapsed("A", lap3_time)
        actual_B3 = get_current_elapsed("B", lap3_time)
        actual_C3 = get_current_elapsed("C", lap3_time)

        delta_A3 = actual_A3 - prev_A2
        delta_B3 = actual_B3 - prev_B2
        delta_C3 = actual_C3 - prev_C2

        assert lap_times_3["A"] == pytest.approx(delta_A3, abs=tol)
        assert lap_times_3["B"] == pytest.approx(delta_B3, abs=tol)
        assert lap_times_3["C"] == pytest.approx(delta_C3, abs=tol)

        # Verify final total elapsed times match our tracking
        for name in ["A", "B", "C"]:
            final_elapsed = stopwatch.get_elapsed(name)
            actual_final = get_current_elapsed(name, lap3_time)
            assert final_elapsed == pytest.approx(actual_final, abs=tol), (
                f"Timer {name}: stopwatch elapsed {final_elapsed} != actual {actual_final}"
            )

        # Build expected checkpoints based on actual measurements
        expected_checkpoints = {
            "A": [0.0, actual_A, actual_A2, actual_A3],
            "B": [0.0, actual_B, actual_B2, actual_B3],
            "C": [0.0, actual_C, actual_C2, actual_C3],
        }

        for name, expected_times in expected_checkpoints.items():
            timer = stopwatch._get_timer(name)
            checkpoints = sorted(timer.checkpoints.items(), key=lambda x: x[1]["elapsed_time"])

            assert len(checkpoints) == len(expected_times), (
                f"Timer {name}: expected {len(expected_times)} checkpoints, got {len(checkpoints)}"
            )

            # Verify checkpoint times match our tracking (with tight tolerance)
            for i, (_, checkpoint_data) in enumerate(checkpoints):
                checkpoint_elapsed = checkpoint_data["elapsed_time"]
                assert checkpoint_elapsed == pytest.approx(expected_times[i], abs=tol), (
                    f"Timer {name}: checkpoint {i} time {checkpoint_elapsed} != expected {expected_times[i]}"
                )

            # Verify lap times match checkpoint deltas
            laps = len(checkpoints) - 1
            for i in range(laps):
                start_checkpoint_time = checkpoints[i][1]["elapsed_time"]
                stop_checkpoint_time = checkpoints[i + 1][1]["elapsed_time"]
                delta_time = stop_checkpoint_time - start_checkpoint_time
                lap_index = -(laps - i)  # Convert forward index to backward index
                lap_time = stopwatch.get_lap_time(lap_index, name)

                assert lap_time is not None, f"Timer {name}: lap {lap_index} does not exist"
                assert abs(delta_time - lap_time) < tol, (
                    f"Timer {name}: checkpoint time {delta_time} != lap time {lap_time} for lap {lap_index}"
                )

    def test_get_lap_steps_first_lap(self, stopwatch: Stopwatch):
        """Test that get_lap_steps correctly handles the first lap.

        Currently, get_lap_steps returns None when there's only one checkpoint,
        but it should return the steps for the first lap (from start to first checkpoint).
        """
        stopwatch.start("test_timer")
        time.sleep(0.1)

        # Record checkpoint at 100 steps
        stopwatch.checkpoint(100, name="test_timer")
        first_lap_steps = stopwatch.get_lap_steps(-1, "test_timer")
        assert first_lap_steps == 100, f"Expected 100 steps for first lap, got {first_lap_steps}"

        # Also test with default parameter (last lap)
        last_lap_steps = stopwatch.get_lap_steps(name="test_timer")
        assert last_lap_steps == 100, f"Expected 100 steps for last lap, got {last_lap_steps}"

        # Add second checkpoint

        time.sleep(0.1)
        stopwatch.checkpoint(250, name="test_timer")

        # Now test both laps
        first_lap_steps = stopwatch.get_lap_steps(-2, "test_timer")
        assert first_lap_steps == 100, f"Expected 100 steps for first lap, got {first_lap_steps}"

        second_lap_steps = stopwatch.get_lap_steps(-1, "test_timer")
        assert second_lap_steps == 150, f"Expected 150 steps for second lap (250-100), got {second_lap_steps}"

        stopwatch.stop("test_timer")


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
        start_time = time.time()

        # First lap: 100 steps
        time.sleep(0.1)
        lap1_end = time.time()
        lap1_time = sw.lap(100, "training")
        lap1_actual = lap1_end - start_time

        assert lap1_time == pytest.approx(lap1_actual, abs=0.02)
        assert lap1_time >= 0.1

        # Second lap: 200 more steps (total 300)
        lap2_start = time.time()
        time.sleep(0.1)
        lap2_end = time.time()
        lap2_time = sw.lap(300, "training")
        lap2_actual = lap2_end - lap2_start

        assert lap2_time == pytest.approx(lap2_actual, abs=0.02)
        assert lap2_time >= 0.1

        # Third lap: 300 more steps (total 600)
        lap3_start = time.time()
        time.sleep(0.1)
        lap3_end = time.time()
        lap3_time = sw.lap(600, "training")
        lap3_actual = lap3_end - lap3_start

        assert lap3_time == pytest.approx(lap3_actual, abs=0.02)
        assert lap3_time >= 0.1

        # Get current lap rate after some more steps
        time.sleep(0.05)
        current_rate = sw.get_lap_rate(650, "training")
        # Rate should be positive and reasonable
        assert current_rate > 0
        assert current_rate < 10000  # Sanity check

        sw.stop("training")

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
        all_elapsed = sw.get_all_elapsed(exclude_global=False)

        # Each component should have minimum expected time
        assert all_elapsed["load_data"] >= 0.05
        assert all_elapsed["process_data"] >= 0.09  # 3 * 0.03
        assert all_elapsed["save_results"] >= 0.02

        # Process should be longest
        assert all_elapsed["process_data"] > all_elapsed["load_data"]
        assert all_elapsed["process_data"] > all_elapsed["save_results"]

        # Total should be at least sum of sleeps
        assert all_elapsed["global"] >= 0.16  # 0.05 + 0.09 + 0.02

    def test_logging_scenarios(self, caplog: pytest.LogCaptureFixture):
        """Test various logging scenarios in integration."""
        sw = Stopwatch(log_level=logging.INFO)

        with caplog.at_level(logging.INFO, logger=sw.logger.name):
            caplog.clear()

            # Test multiple operations with logging
            with sw("operation1", log_level=logging.INFO):
                time.sleep(0.05)

            with sw("operation2", log_level=logging.INFO):
                time.sleep(0.03)

            # Test progress logging
            sw.start("batch_process")
            for i in range(3):
                time.sleep(0.02)
                sw.log_progress((i + 1) * 33, 100, "batch_process", "Batch Processing")
            sw.stop("batch_process")

            # Verify all expected logs
            info_records = [r for r in caplog.records if r.levelname == "INFO" and r.name == sw.logger.name]

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

    def test_timer_decorators(self, caplog: pytest.LogCaptureFixture):
        """Test both @with_timer and @with_instance_timer decorators."""

        sw = Stopwatch()

        class TestClass:
            def __init__(self):
                self.timer = Stopwatch(log_level=logging.INFO)  # Default timer attribute with logging
                self.instance_timer = Stopwatch(log_level=logging.INFO)  # Custom timer attribute with logging
                self.call_count = 0

            @with_timer(sw, "external_timer")
            def external_timed_method(self, value: int) -> int:
                """Method timed with external timer."""
                self.call_count += 1
                time.sleep(0.05)
                return value * 2

            @with_instance_timer("instance_timer_default")
            def instance_timed_method(self, value: int) -> int:
                """Method timed with instance timer (default attr)."""
                time.sleep(0.03)
                return value + 1

            @with_instance_timer("custom_timer", timer_attr="instance_timer")
            def custom_attr_method(self, value: int) -> int:
                """Method timed with custom timer attribute."""
                time.sleep(0.02)
                return value * 3

            @with_instance_timer("logged_method", log_level=logging.INFO)
            def logged_method(self, x: int, y: int = 10) -> int:
                """Method with logging enabled."""
                time.sleep(0.03)
                return x + y

        test_obj = TestClass()

        # Test external timer decorator
        result = test_obj.external_timed_method(5)
        assert result == 10
        assert test_obj.call_count == 1

        external_elapsed = sw.get_elapsed("external_timer")
        assert external_elapsed == pytest.approx(0.05, abs=0.1)

        # Test instance timer decorator (default attribute)
        result = test_obj.instance_timed_method(5)
        assert result == 6

        instance_elapsed = test_obj.timer.get_elapsed("instance_timer_default")
        assert instance_elapsed == pytest.approx(0.03, abs=0.1)

        # Test instance timer with custom attribute
        result = test_obj.custom_attr_method(5)
        assert result == 15

        custom_elapsed = test_obj.instance_timer.get_elapsed("custom_timer")
        assert custom_elapsed == pytest.approx(0.02, abs=0.1)

        # Test logging functionality
        with caplog.at_level(logging.INFO, logger=sw.logger.name):
            caplog.clear()

            result = test_obj.logged_method(5, y=15)
            assert result == 20

            # Check log output
            instance_timer_logs = [r for r in caplog.records if r.name == test_obj.timer.logger.name]

            assert len(instance_timer_logs) == 1
            assert instance_timer_logs[0].levelname == "INFO"
            assert "logged_method took" in instance_timer_logs[0].message

        # Test multiple calls accumulate time
        test_obj.external_timed_method(3)
        total_external_elapsed = sw.get_elapsed("external_timer")
        assert total_external_elapsed > external_elapsed

        # Test function metadata preservation
        assert test_obj.external_timed_method.__name__ == "external_timed_method"
        assert test_obj.external_timed_method.__doc__ is not None
        assert "Method timed with external timer." in test_obj.external_timed_method.__doc__

        assert test_obj.instance_timed_method.__name__ == "instance_timed_method"
        assert test_obj.instance_timed_method.__doc__ is not None
        assert "Method timed with instance timer" in test_obj.instance_timed_method.__doc__

        # Test exception handling for both decorators
        @with_timer(sw, "exception_timer")
        def failing_external():
            time.sleep(0.02)
            raise ValueError("External exception")

        with pytest.raises(ValueError, match="External exception"):
            failing_external()

        exception_elapsed = sw.get_elapsed("exception_timer")
        assert exception_elapsed == pytest.approx(0.02, abs=0.1)

        # Test instance timer exception handling
        class ExceptionTestClass:
            def __init__(self):
                self.timer = Stopwatch()

            @with_instance_timer("exception_instance_timer")
            def failing_instance_method(self):
                time.sleep(0.02)
                raise RuntimeError("Instance exception")

        exception_obj = ExceptionTestClass()
        with pytest.raises(RuntimeError, match="Instance exception"):
            exception_obj.failing_instance_method()

        instance_exception_elapsed = exception_obj.timer.get_elapsed("exception_instance_timer")
        assert instance_exception_elapsed == pytest.approx(0.02, abs=0.1)

        # Test error case: with_instance_timer on non-instance method
        with pytest.raises(ValueError, match="with_instance_timer can only be used on instance methods"):

            @with_instance_timer("standalone_timer")
            def standalone_function():
                pass

            standalone_function()  # This should raise because no 'self' argument

    def test_decorator_edge_cases(self):
        """Test edge cases for decorators."""

        # Test missing timer attribute
        class MissingTimerClass:
            def __init__(self):
                pass  # No timer attribute

            @with_instance_timer("test_timer")
            def method_without_timer(self):
                return "test"

        obj = MissingTimerClass()
        with pytest.raises(AttributeError):
            obj.method_without_timer()

        # Test custom timer attribute name that doesn't exist
        class CustomTimerClass:
            def __init__(self):
                self.timer = Stopwatch()  # Has 'timer' but decorator looks for 'custom_timer'

            @with_instance_timer("test_timer", timer_attr="custom_timer")
            def method_with_missing_custom_attr(self):
                return "test"

        obj2 = CustomTimerClass()
        with pytest.raises(AttributeError):
            obj2.method_with_missing_custom_attr()


class TestStopwatchSaveLoad:
    """Test save/load functionality of Stopwatch."""

    def test_save_load_basic(self, stopwatch: Stopwatch):
        """Test basic save and load functionality."""
        # Create some timer state
        stopwatch.start("timer1")
        time.sleep(0.1)
        stopwatch.stop("timer1")

        stopwatch.start("timer2")
        time.sleep(0.05)
        stopwatch.checkpoint(100, "checkpoint1", "timer2")
        stopwatch.stop("timer2")

        # Save state
        state = stopwatch.save_state()

        # Verify state structure
        assert "version" in state
        assert state["version"] == "1.0"
        assert "timers" in state
        assert len(state["timers"]) >= 2  # At least timer1, timer2 (and global)

        # Create new stopwatch and load state
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state)

        # Verify timers were restored
        assert new_stopwatch.get_elapsed("timer1") == pytest.approx(stopwatch.get_elapsed("timer1"), abs=0.1)
        assert new_stopwatch.get_elapsed("timer2") == pytest.approx(stopwatch.get_elapsed("timer2"), abs=0.1)

        # Verify checkpoint was restored
        timer2 = new_stopwatch._get_timer("timer2")
        assert "checkpoint1" in timer2.checkpoints
        assert timer2.checkpoints["checkpoint1"]["steps"] == 100

    def test_save_load_running_timers(self, stopwatch: Stopwatch):
        """Test saving and loading with running timers."""
        # Start multiple timers
        stopwatch.start("running1")
        time.sleep(0.1)

        stopwatch.start("running2")
        time.sleep(0.05)

        stopwatch.start("stopped1")
        time.sleep(0.05)
        stopwatch.stop("stopped1")

        # Get original elapsed times for running timers
        orig_elapsed1 = stopwatch.get_elapsed("running1")
        orig_elapsed2 = stopwatch.get_elapsed("running2")

        # Save state while timers are running
        state = stopwatch.save_state()

        # Verify running timers are marked
        assert state["timers"]["running1"]["_was_running"] is True
        assert state["timers"]["running2"]["_was_running"] is True
        assert state["timers"]["stopped1"]["_was_running"] is False

        # Load with resume_running=True (default)
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state, resume_running=True)

        # Verify timers are running
        assert new_stopwatch._get_timer("running1").is_running()
        assert new_stopwatch._get_timer("running2").is_running()
        assert not new_stopwatch._get_timer("stopped1").is_running()

        # Let them run a bit more
        time.sleep(0.1)

        # Stop and check elapsed times are greater than original
        new_stopwatch.stop("running1")
        new_stopwatch.stop("running2")

        assert new_stopwatch.get_elapsed("running1") > orig_elapsed1 + 0.09
        assert new_stopwatch.get_elapsed("running2") > orig_elapsed2 + 0.09

    def test_save_load_no_resume(self, stopwatch: Stopwatch):
        """Test loading with resume_running=False."""
        # Start a timer
        stopwatch.start("test_timer")
        time.sleep(0.1)

        # Save state
        state = stopwatch.save_state()

        # Load without resuming
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state, resume_running=False)

        # Timer should not be running
        assert not new_stopwatch._get_timer("test_timer").is_running()

        # Elapsed time should match what was saved
        saved_elapsed = state["timers"]["test_timer"]["total_elapsed"]
        assert new_stopwatch.get_elapsed("test_timer") == pytest.approx(saved_elapsed, abs=0.1)

    def test_save_load_complex_state(self, stopwatch: Stopwatch):
        """Test saving/loading complex timer state with multiple features."""
        # Create complex state
        stopwatch.start("complex_timer")
        time.sleep(0.05)
        stopwatch.checkpoint(100, "check1", "complex_timer")
        time.sleep(0.05)
        stopwatch.checkpoint(200, "check2", "complex_timer")
        time.sleep(0.05)
        # Don't use lap() as it creates an auto checkpoint
        stopwatch.stop("complex_timer")

        # Add references manually for testing
        timer = stopwatch._get_timer("complex_timer")
        # Note: start() already added one reference, so we'll have 3 total
        timer.references.add(("test1.py", 10))
        timer.references.add(("test2.py", 20))

        # Save state
        state = stopwatch.save_state()

        # Load into new stopwatch
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state)

        # Verify all aspects were preserved
        new_timer = new_stopwatch._get_timer("complex_timer")

        # Check elapsed time
        assert new_stopwatch.get_elapsed("complex_timer") == pytest.approx(
            stopwatch.get_elapsed("complex_timer"), abs=0.1
        )

        # Check checkpoints
        assert len(new_timer.checkpoints) == 3  # _start, check1, check2
        assert "check1" in new_timer.checkpoints
        assert "check2" in new_timer.checkpoints
        assert new_timer.checkpoints["check1"]["steps"] == 100
        assert new_timer.checkpoints["check2"]["steps"] == 200

        # Check lap counter
        assert new_timer.lap_counter == timer.lap_counter

        # Check references - should have 3 (1 from start + 2 manual)
        assert len(new_timer.references) == 3
        # The first reference is from start(), check the manually added ones
        assert ("test1.py", 10) in new_timer.references
        assert ("test2.py", 20) in new_timer.references

    def test_save_load_empty_state(self):
        """Test saving/loading empty stopwatch."""
        sw = Stopwatch()

        # Save empty state (only has global timer)
        state = sw.save_state()

        # Load into new stopwatch
        new_sw = Stopwatch()
        new_sw.load_state(state)

        # Should have global timer
        assert new_sw.GLOBAL_TIMER_NAME in new_sw._timers
        assert len(new_sw._timers) >= 1

    def test_load_invalid_state(self, stopwatch: Stopwatch):
        """Test loading invalid state formats."""
        # Test completely invalid state
        with pytest.raises(ValueError, match="Invalid state format"):
            stopwatch.load_state("not a dict")  # type: ignore

        # Test missing timers key
        with pytest.raises(ValueError, match="Invalid state format"):
            stopwatch.load_state({"version": "1.0"})

        # Test empty dict
        with pytest.raises(ValueError, match="Invalid state format"):
            stopwatch.load_state({})

    def test_save_load_preserves_global_timer(self, stopwatch: Stopwatch):
        """Test that global timer is always preserved."""
        # Start global timer
        stopwatch.start()
        time.sleep(0.1)
        stopwatch.stop()

        # Save and load
        state = stopwatch.save_state()
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state)

        # Check global timer exists and has correct elapsed time
        assert new_stopwatch.GLOBAL_TIMER_NAME in new_stopwatch._timers
        assert new_stopwatch.get_elapsed() == pytest.approx(stopwatch.get_elapsed(), abs=0.1)

    def test_elapsed_time_accuracy_with_running_timers(self, stopwatch: Stopwatch):
        """Test that elapsed time is accurately preserved for running timers."""
        # Start timer and let it run
        stopwatch.start("accuracy_test")
        time.sleep(0.2)

        # Get elapsed before save
        elapsed_before_save = stopwatch.get_elapsed("accuracy_test")

        # Save while running
        state = stopwatch.save_state()

        # The saved state should include elapsed time up to save point
        saved_elapsed = state["timers"]["accuracy_test"]["total_elapsed"]
        assert saved_elapsed > 0.19
        assert saved_elapsed == pytest.approx(elapsed_before_save, abs=0.01)

        # Load immediately
        new_stopwatch = Stopwatch()
        new_stopwatch.load_state(state, resume_running=True)

        # Stop immediately and check elapsed time
        new_stopwatch.stop("accuracy_test")

        # Total elapsed should be close to saved elapsed (plus tiny bit for stop operation)
        assert new_stopwatch.get_elapsed("accuracy_test") == pytest.approx(saved_elapsed, abs=0.01)

    def test_multiple_save_load_cycles(self, stopwatch: Stopwatch):
        """Test multiple save/load cycles maintain accuracy."""
        # Initial timer
        stopwatch.start("cycle_test")
        time.sleep(0.1)
        stopwatch.checkpoint(100, "checkpoint1", "cycle_test")

        # Get elapsed after first checkpoint
        elapsed1 = stopwatch.get_elapsed("cycle_test")

        # First save/load cycle
        state1 = stopwatch.save_state()
        sw2 = Stopwatch()
        sw2.load_state(state1, resume_running=True)

        time.sleep(0.1)
        sw2.checkpoint(200, "checkpoint2", "cycle_test")

        # Get elapsed after second checkpoint
        elapsed2 = sw2.get_elapsed("cycle_test")
        assert elapsed2 > elapsed1 + 0.09

        # Second save/load cycle
        state2 = sw2.save_state()
        sw3 = Stopwatch()
        sw3.load_state(state2, resume_running=True)

        time.sleep(0.1)
        sw3.stop("cycle_test")

        # Final timer should have all checkpoints and correct elapsed time
        timer = sw3._get_timer("cycle_test")
        assert len(timer.checkpoints) == 3  # with _start
        assert "checkpoint1" in timer.checkpoints
        assert "checkpoint2" in timer.checkpoints

        # Total elapsed should be at least 0.3 seconds
        final_elapsed = sw3.get_elapsed("cycle_test")
        assert final_elapsed > 0.29

    def test_load_clears_existing_timers(self, stopwatch: Stopwatch):
        """Test that loading state clears existing timers."""
        # Create initial timers
        stopwatch.start("existing1")
        time.sleep(0.05)
        stopwatch.stop("existing1")

        stopwatch.start("existing2")
        time.sleep(0.05)
        stopwatch.stop("existing2")

        # Create new state with different timers
        new_sw = Stopwatch()
        new_sw.start("new1")
        time.sleep(0.05)
        new_sw.stop("new1")

        state = new_sw.save_state()

        # Load new state into original stopwatch
        stopwatch.load_state(state)

        # Old timers should be gone
        assert "existing1" not in stopwatch._timers
        assert "existing2" not in stopwatch._timers

        # New timer should exist
        assert "new1" in stopwatch._timers
        assert stopwatch.get_elapsed("new1") > 0.04


def test_cleanup_old_checkpoints_preserves_order():
    """Test that _cleanup_old_checkpoints maintains chronological order."""

    stopwatch = Stopwatch(max_laps=3)
    timer_name = "test_timer"

    # Create a timer and add checkpoints in chronological order
    timer = stopwatch._get_timer(timer_name)

    # Manually add checkpoints to ensure we know the exact order and timing
    checkpoints_data = [
        ("checkpoint_1", 1.0, 100),
        ("checkpoint_2", 2.0, 200),
        ("checkpoint_3", 3.0, 300),
        ("checkpoint_4", 4.0, 400),
        ("checkpoint_5", 5.0, 500),
    ]

    # Add all checkpoints
    for name, elapsed, steps in checkpoints_data:
        timer.checkpoints[name] = Checkpoint(elapsed_time=elapsed, steps=steps)

    # Verify initial state - should have 6 checkpoints with "_start"
    assert len(timer.checkpoints) == 6

    # Get the order before cleanup
    before_cleanup = list(timer.checkpoints.items())
    print("Before cleanup:")
    for name, cp in before_cleanup:
        print(f"  {name}: elapsed={cp['elapsed_time']}, steps={cp['steps']}")

    # Trigger cleanup (max_laps=3 means max_checkpoints=4, so 6 > 4 triggers cleanup)
    timer.cleanup_old_checkpoints()

    # Should now have 4 checkpoints (max_laps + 1)
    assert len(timer.checkpoints) == 4

    # Get the order after cleanup
    after_cleanup = list(timer.checkpoints.items())
    print("\nAfter cleanup:")
    for name, cp in after_cleanup:
        print(f"  {name}: elapsed={cp['elapsed_time']}, steps={cp['steps']}")

    # Verify that we kept the LAST 4 checkpoints
    expected_remaining = checkpoints_data[-4:]  # Last 4 items

    actual_items = list(timer.checkpoints.items())

    for i, (expected_name, expected_elapsed, expected_steps) in enumerate(expected_remaining):
        actual_name, actual_checkpoint = actual_items[i]

        assert actual_name == expected_name, f"Name mismatch at index {i}: expected {expected_name}, got {actual_name}"
        assert actual_checkpoint["elapsed_time"] == expected_elapsed, f"Elapsed time mismatch at index {i}"
        assert actual_checkpoint["steps"] == expected_steps, f"Steps mismatch at index {i}"

    # Verify chronological order is maintained (elapsed_time should be increasing)
    elapsed_times = [cp["elapsed_time"] for cp in timer.checkpoints.values()]
    assert elapsed_times == sorted(elapsed_times), f"Checkpoints not in chronological order: {elapsed_times}"

    # Verify step counts are increasing
    step_counts = [cp["steps"] for cp in timer.checkpoints.values()]
    assert step_counts == sorted(step_counts), f"Step counts not in increasing order: {step_counts}"


def test_multiple_laps_after_cleanup():
    """Test multiple lap indices after cleanup."""

    stopwatch = Stopwatch(max_laps=2)  # Very restrictive to force cleanup

    timer = stopwatch._get_timer()

    # Add more checkpoints than max_laps allows
    checkpoints_data = [
        ("_lap_1", 1.0, 100),
        ("_lap_2", 2.0, 250),  # 150 step lap
        ("_lap_3", 3.0, 400),  # 150 step lap
        ("_lap_4", 4.0, 600),  # 200 step lap
    ]

    for name, elapsed, steps in checkpoints_data:
        timer.checkpoints[name] = Checkpoint(elapsed_time=elapsed, steps=steps)
        timer.lap_counter += 1

    print("Before cleanup - all checkpoints:")
    for name, cp in timer.checkpoints.items():
        print(f"  {name}: elapsed={cp['elapsed_time']}, steps={cp['steps']}")

    # Force cleanup (max_laps=2 means keep 3 checkpoints)
    timer.cleanup_old_checkpoints()

    print("\nAfter cleanup:")
    for name, cp in timer.checkpoints.items():
        print(f"  {name}: elapsed={cp['elapsed_time']}, steps={cp['steps']}")

    # Now test lap calculations
    print("\nLap calculations:")

    # Most recent lap (-1): should be lap_4 - lap_3 = 600 - 400 = 200
    lap_1 = stopwatch.get_lap_steps(-1)
    print(f"Lap -1 (most recent): {lap_1}, expected: 200")

    # Second most recent lap (-2): should be lap_3 - lap_2 = 400 - 250 = 150
    lap_2 = stopwatch.get_lap_steps(-2)
    print(f"Lap -2 (second recent): {lap_2}, expected: 150")

    # This should fail because we don't have enough checkpoints
    lap_3 = stopwatch.get_lap_steps(-3)
    print(f"Lap -3 (should be None): {lap_3}")

    assert lap_1 == 200, f"Expected lap -1 to be 200, got {lap_1}"
    assert lap_2 == 150, f"Expected lap -2 to be 150, got {lap_2}"
    assert lap_3 is None, f"Expected lap -3 to be None, got {lap_3}"

    # now we will add some laps using the timer
    stopwatch.start()
    time.sleep(0.05)
    stopwatch.lap_all(steps=850, exclude_global=False)
    time.sleep(0.05)
    stopwatch.lap_all(steps=1150, exclude_global=False)
    stopwatch.stop()

    print("\nAfter laps:")
    for name, cp in timer.checkpoints.items():
        print(f"  {name}: elapsed={cp['elapsed_time']}, steps={cp['steps']}")

    lap_1 = stopwatch.get_lap_steps(-1)
    print(f"Lap -1 (most recent): {lap_1}, expected: 300")

    lap_2 = stopwatch.get_lap_steps(-2)
    print(f"Lap -2 (second recent): {lap_2}, expected: 250")

    lap_3 = stopwatch.get_lap_steps(-3)
    print(f"Lap -3 (should be None): {lap_3}")

    assert lap_1 == 300, f"Expected lap -1 to be 300, got {lap_1}"
    assert lap_2 == 250, f"Expected lap -2 to be 250, got {lap_2}"
    assert lap_3 is None, f"Expected lap -3 to be None, got {lap_3}"
