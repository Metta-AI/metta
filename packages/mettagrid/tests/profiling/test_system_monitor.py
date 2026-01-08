import logging
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
