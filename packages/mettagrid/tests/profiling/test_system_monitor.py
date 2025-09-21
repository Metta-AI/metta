import logging
import time
from typing import Generator

import pytest

from mettagrid.profiling.system_monitor import SystemMonitor


@pytest.fixture
def monitor() -> Generator[SystemMonitor, None, None]:
    """Create a SystemMonitor instance with auto_start=False"""
    monitor = SystemMonitor(sampling_interval_sec=0.1, history_size=10, auto_start=False, log_level=logging.INFO)
    yield monitor
    # Cleanup
    monitor.stop()


class TestInitialization:
    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        monitor = SystemMonitor(sampling_interval_sec=2.0, history_size=50, log_level=logging.DEBUG, auto_start=False)

        assert monitor.sampling_interval_sec == 2.0
        assert monitor.history_size == 50
        assert monitor.logger.level == logging.DEBUG

    def test_auto_start(self):
        """Test auto_start functionality"""
        monitor = SystemMonitor(sampling_interval_sec=0.1, auto_start=True)

        # Should be running
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        # Clean up
        monitor.stop()


class TestMonitoringControl:
    def test_start_stop(self, monitor):
        """Test starting and stopping monitoring"""
        # Initially not running
        assert not monitor._thread or not monitor._thread.is_alive()

        # Start monitoring
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        # Stop monitoring
        monitor.stop()

        # Allow the background thread to finish
        deadline = time.perf_counter() + 1.0
        while monitor._thread.is_alive() and time.perf_counter() < deadline:
            time.sleep(0.02)

        assert not monitor._thread.is_alive()

    def test_double_start(self, monitor, caplog):
        """Test starting when already running"""
        monitor.start()

        # Capture logs from the specific logger
        with caplog.at_level(logging.WARNING, logger=monitor.logger.name):
            monitor.start()

        assert "Monitor already running" in caplog.text
        monitor.stop()

    def test_stop_when_not_running(self, monitor):
        """Test stopping when not running (should not raise)"""
        # Should not raise any exception
        monitor.stop()

    def test_thread_collection(self, monitor):
        """Test that monitoring thread collects samples"""
        monitor.start()

        # Wait for a few samples
        time.sleep(0.3)

        # Check that metrics have been collected (internal state)
        assert len(monitor._metrics) > 0
        assert len(monitor.get_latest()) > 0

        monitor.stop()


class TestInternalState:
    """Test internal state management (since public accessors were removed)"""

    def test_metrics_initialized(self, monitor):
        """Test that metrics collectors are initialized"""
        assert len(monitor._metric_collectors) > 0
        assert "cpu_percent" in monitor._metric_collectors
        assert "memory_percent" in monitor._metric_collectors

    def test_collect_sample(self, monitor):
        """Test internal sample collection"""
        # This tests internal implementation
        monitor._collect_sample()

        # Check that latest values were recorded
        assert len(monitor.get_latest()) > 0
        assert "cpu_percent" in monitor.get_latest()
        assert "memory_percent" in monitor.get_latest()

        # Check that values are reasonable
        cpu_percent = monitor.get_latest().get("cpu_percent")
        assert cpu_percent is not None
        assert 0 <= cpu_percent <= 100 * monitor.get_latest().get("cpu_count", 1)

        memory_percent = monitor.get_latest().get("memory_percent")
        assert memory_percent is not None
        assert 0 <= memory_percent <= 100
