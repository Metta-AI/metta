import logging
import platform
import time
from collections import deque
from typing import Generator

import psutil
import pytest
import torch

from metta.common.util.system_monitor import SystemMonitor

# Platform detection helpers
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


# Test doubles and helpers
class FakeProcess:
    """Simple test double for psutil.Process"""

    def __init__(self, memory_rss=100 * 1024 * 1024, cpu_percent=25.0, num_threads=4):
        self._memory_rss = memory_rss
        self._cpu_percent = cpu_percent
        self._num_threads = num_threads
        self._first_call = True

    def memory_info(self):
        return type("MemInfo", (), {"rss": self._memory_rss})()

    def cpu_percent(self, interval=None):
        # Simulate the behavior where first call might return 0
        if self._first_call and interval is None:
            self._first_call = False
            return 0.0
        return self._cpu_percent

    def num_threads(self):
        return self._num_threads


class FakeVirtualMemory:
    """Simple test double for psutil virtual memory"""

    def __init__(self, total=8192, available=4096, used=4096, percent=50.0):
        self.total = total * 1024 * 1024  # Convert to bytes
        self.available = available * 1024 * 1024
        self.used = used * 1024 * 1024
        self.percent = percent


# Fixtures
@pytest.fixture
def monitor() -> Generator[SystemMonitor, None, None]:
    """Create a SystemMonitor instance with auto_start=False"""
    monitor = SystemMonitor(sampling_interval_sec=0.1, history_size=10, auto_start=False)
    yield monitor
    # Cleanup
    monitor.stop()


@pytest.fixture
def mock_psutil(monkeypatch) -> dict:
    """Mock psutil functions with controllable values, platform-aware"""
    mock_data = {
        "cpu_percent": 50.0,
        "cpu_count": 8,
        "cpu_count_logical": 8,
        "cpu_count_physical": 4,
        "memory": FakeVirtualMemory(),
        "process": FakeProcess(),
        "temperatures": {},
    }

    def mock_cpu_percent(interval=None):
        return mock_data["cpu_percent"]

    def mock_cpu_count(logical=True):
        if logical:
            return mock_data["cpu_count_logical"]
        return mock_data["cpu_count_physical"]

    def mock_virtual_memory():
        return mock_data["memory"]

    def mock_process(pid=None):
        # Return the same instance to simulate persistent process object
        if not hasattr(mock_process, "_instance"):
            mock_process._instance = mock_data["process"]  # type: ignore
        return mock_process._instance  # type: ignore

    def mock_sensors_temperatures():
        return mock_data["temperatures"]

    monkeypatch.setattr(psutil, "cpu_percent", mock_cpu_percent)
    monkeypatch.setattr(psutil, "cpu_count", mock_cpu_count)
    monkeypatch.setattr(psutil, "virtual_memory", mock_virtual_memory)
    monkeypatch.setattr(psutil, "Process", mock_process)

    # Only mock sensors_temperatures if the platform typically has it
    # macOS typically doesn't have sensors_temperatures
    if not IS_MACOS or hasattr(psutil, "sensors_temperatures"):
        monkeypatch.setattr(psutil, "sensors_temperatures", mock_sensors_temperatures)

    return mock_data


@pytest.fixture
def mock_torch_cuda(monkeypatch) -> dict | None:
    """Mock torch CUDA functions - only if CUDA is available on the platform"""
    if not HAS_CUDA:
        # Don't mock CUDA if it's not available
        return None

    mock_data = {
        "is_available": True,
        "device_count": 2,
        "utilization": [60.0, 40.0],
        "memory_info": [(2 * 1024**3, 8 * 1024**3), (3 * 1024**3, 8 * 1024**3)],  # (free, total)
    }

    def mock_is_available():
        return mock_data["is_available"]

    def mock_device_count():
        return mock_data["device_count"]

    def mock_utilization(device):
        return mock_data["utilization"][device]

    def mock_mem_get_info(device):
        return mock_data["memory_info"][device]

    monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)
    monkeypatch.setattr(torch.cuda, "device_count", mock_device_count)
    monkeypatch.setattr(torch.cuda, "utilization", mock_utilization)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mock_mem_get_info)

    return mock_data


# Tests for initialization
class TestInitialization:
    def test_init_default_params(self, mock_psutil):
        """Test initialization with default parameters"""
        monitor = SystemMonitor(auto_start=False)

        assert monitor.sampling_interval_sec == 1.0
        assert monitor.history_size == 100
        assert not monitor._thread or not monitor._thread.is_alive()
        assert len([v for v in monitor._metric_collectors.values() if v]) > 0

    def test_init_custom_params(self, mock_psutil):
        """Test initialization with custom parameters"""
        monitor = SystemMonitor(sampling_interval_sec=2.0, history_size=50, auto_start=False)

        assert monitor.sampling_interval_sec == 2.0
        assert monitor.history_size == 50

    def test_custom_logger(self, mock_psutil):
        """Test initialization with custom logger"""
        custom_logger = logging.getLogger("test_logger")
        monitor = SystemMonitor(logger=custom_logger, auto_start=False)

        assert monitor.logger == custom_logger


class TestMetricCollection:
    """Tests for basic metric collection functionality"""

    def test_cpu_metrics(self, monitor, mock_psutil):
        """Test CPU metric collection"""
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert latest["cpu_percent"] == 50.0
        assert latest["cpu_count"] == 8
        assert latest["cpu_count_logical"] == 8
        assert latest["cpu_count_physical"] == 4

    def test_process_cpu_initialization(self, mock_psutil):
        """Test that process CPU is properly initialized"""
        # Create monitor which should initialize process CPU
        monitor = SystemMonitor(auto_start=False)

        # First collection might be 0 due to initialization
        monitor._collect_sample()

        # Second collection should have real value
        monitor._collect_sample()
        latest = monitor._latest

        # Should have non-zero CPU after initialization
        assert latest["process_cpu_percent"] == 25.0

    def test_memory_metrics(self, monitor, mock_psutil):
        """Test memory metric collection"""
        monitor._collect_sample()

        latest = monitor._latest
        assert latest["memory_percent"] == 50.0
        assert latest["memory_available_mb"] == 4096.0
        assert latest["memory_used_mb"] == 4096.0
        assert latest["memory_total_mb"] == 8192.0

    def test_process_metrics(self, monitor, mock_psutil):
        """Test process-specific metric collection"""
        import time

        # Note: The monitor uses a persistent process object created during initialization,
        # which tracks the real system process, not our mock. The mock only affects
        # new Process() calls for memory and thread metrics.

        # Warm up the CPU measurement - first call often returns 0 or unreliable values
        monitor._collect_sample()

        # Add a small delay to ensure meaningful CPU measurement interval
        time.sleep(0.1)

        # Second sample should have actual values
        monitor._collect_sample()
        latest = monitor.get_latest()

        # Check memory and threads (these create new Process objects which use our mock)
        assert latest["process_memory_mb"] == 100.0
        assert latest["process_threads"] == 4

        # Process CPU uses the persistent _process object created during __init__
        # This is the REAL system process, not our mock, so we can't predict its exact value
        assert isinstance(latest["process_cpu_percent"], (int, float))
        assert latest["process_cpu_percent"] >= 0

        # CPU percentage can exceed 100% for multi-threaded processes
        # Just ensure it's within a sanity range (e.g., not 8000%+)
        assert latest["process_cpu_percent"] < 10000, (
            f"Process CPU percent {latest['process_cpu_percent']} is unreasonably high, "
            f"likely indicates a measurement error"
        )

    def test_temperature_metrics(self, monitor, mock_psutil):
        """Test CPU temperature collection"""
        # Skip on platforms without temperature sensors
        if not hasattr(psutil, "sensors_temperatures"):
            pytest.skip("Temperature sensors not available on this platform")

        # Add temperature data
        mock_psutil["temperatures"] = {"coretemp": [type("Temp", (), {"current": 65.0})()]}

        # Reinitialize to pick up temperature metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        latest = monitor.get_latest()
        # Only assert if the metric was actually added
        if "cpu_temperature" in monitor.get_available_metrics():
            assert latest.get("cpu_temperature") == 65.0
        else:
            # Platform doesn't support temperature monitoring
            assert "cpu_temperature" not in latest

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_gpu_metrics_cuda(self, monitor, mock_psutil, mock_torch_cuda):
        """Test GPU metric collection with CUDA"""
        # Reinitialize to pick up GPU metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        latest = monitor.get_latest()

        # Test aggregate metrics with new names
        assert latest["gpu_count"] == 2
        assert latest["gpu_utilization_avg"] == 50.0  # Average of 60 and 40
        assert latest["gpu_memory_percent_avg"] == 68.75  # Average of 75% and 62.5%
        assert latest["gpu_memory_used_mb_total"] == 11264.0  # Total used across GPUs

        # Test per-GPU metrics
        assert latest["gpu0_utilization"] == 60.0
        assert latest["gpu0_memory_percent"] == 75.0
        assert latest["gpu0_memory_used_mb"] == 6144.0

        assert latest["gpu1_utilization"] == 40.0
        assert latest["gpu1_memory_percent"] == 62.5
        assert latest["gpu1_memory_used_mb"] == 5120.0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_per_gpu_metrics(self, monitor, mock_psutil, mock_torch_cuda):
        """Test per-GPU metric collection"""
        # Reinitialize to pick up GPU metrics
        monitor._initialize_default_metrics()

        # Verify all per-GPU metrics are registered
        metrics = monitor.get_available_metrics()
        for i in range(2):  # mock has 2 GPUs
            assert f"gpu{i}_utilization" in metrics
            assert f"gpu{i}_memory_percent" in metrics
            assert f"gpu{i}_memory_used_mb" in metrics

        # Collect samples
        monitor._collect_sample()
        latest = monitor.get_latest()

        # Verify values match mock data
        assert latest["gpu0_utilization"] == 60.0
        assert latest["gpu1_utilization"] == 40.0


# Tests for monitoring control
class TestMonitoringControl:
    def test_start_stop(self, monitor, mock_psutil):
        """Test starting and stopping monitoring with a brief teardown wait."""
        assert not monitor.is_running()

        monitor.start()
        assert monitor.is_running()

        monitor.stop()

        # Allow the background thread to finish (max ~1s)
        deadline = time.perf_counter() + 1.0
        while monitor.is_running() and time.perf_counter() < deadline:
            time.sleep(0.02)

        assert not monitor.is_running()

    def test_double_start(self, monitor, mock_psutil, caplog):
        """Test starting when already running"""
        monitor.start()

        with caplog.at_level(logging.WARNING):
            monitor.start()

        assert "Monitor already running" in caplog.text
        monitor.stop()

    def test_thread_collection(self, monitor, mock_psutil):
        """Test that monitoring thread collects samples"""
        monitor.start()

        # Wait for a few samples
        time.sleep(0.3)

        history = monitor.get_history("cpu_percent")
        assert len(history) >= 2

        monitor.stop()


# Tests for data retrieval
class TestDataRetrieval:
    def test_get_latest_single_metric(self, monitor, mock_psutil):
        """Test getting latest value for single metric"""
        monitor._collect_sample()

        cpu_percent = monitor.get_latest("cpu_percent")
        assert cpu_percent == 50.0

    def test_get_latest_all_metrics(self, monitor, mock_psutil):
        """Test getting all latest values"""
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert isinstance(latest, dict)
        assert "cpu_percent" in latest
        assert "memory_percent" in latest

    def test_get_latest_nonexistent_metric(self, monitor, mock_psutil):
        """Test getting latest value for non-existent metric"""
        monitor._collect_sample()

        value = monitor.get_latest("nonexistent_metric")
        assert value is None

    def test_get_history(self, monitor, mock_psutil):
        """Test getting metric history"""
        # Collect multiple samples
        for i in range(5):
            mock_psutil["cpu_percent"] = 40.0 + i * 5
            monitor._collect_sample()
            time.sleep(0.01)

        history = monitor.get_history("cpu_percent")
        assert len(history) == 5

        # Check structure and values
        for i, (timestamp, value) in enumerate(history):
            assert isinstance(timestamp, float)
            assert value == 40.0 + i * 5

    def test_get_history_nonexistent_metric(self, monitor, mock_psutil):
        """Test getting history for non-existent metric"""
        history = monitor.get_history("nonexistent_metric")
        assert history == []

    def test_history_size_limit(self, monitor, mock_psutil):
        """Test that history respects size limit"""
        # Collect more samples than history_size
        for _ in range(15):
            monitor._collect_sample()

        history = monitor.get_history("cpu_percent")
        assert len(history) == 10  # history_size from fixture


# Tests for summary and reporting
class TestSummaryAndReporting:
    def test_get_summary(self, monitor, mock_psutil):
        """Test getting comprehensive summary"""
        # Collect samples with varying values
        cpu_values = [30.0, 50.0, 70.0, 40.0, 60.0]
        for val in cpu_values:
            mock_psutil["cpu_percent"] = val
            monitor._collect_sample()

        summary = monitor.get_summary()

        assert "timestamp" in summary
        assert "metrics" in summary

        cpu_stats = summary["metrics"]["cpu_percent"]
        assert cpu_stats["latest"] == 60.0
        assert cpu_stats["average"] == 50.0
        assert cpu_stats["min"] == 30.0
        assert cpu_stats["max"] == 70.0
        assert cpu_stats["sample_count"] == 5

    def test_get_available_metrics(self, monitor, mock_psutil):
        """Test getting list of available metrics"""
        metrics = monitor.get_available_metrics()

        assert isinstance(metrics, list)
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "process_memory_mb" in metrics

    def test_log_summary(self, monitor, mock_psutil, caplog):
        """Test log summary output"""
        monitor._collect_sample()

        with caplog.at_level(logging.INFO):
            monitor.log_summary()

        assert "System Monitor Summary" in caplog.text
        assert "cpu_percent" in caplog.text
        assert "memory_percent" in caplog.text


# Tests for context manager
class TestContextManager:
    def test_monitor_context(self, monitor, mock_psutil, caplog):
        """Test monitor context manager"""
        monitor._collect_sample()

        with caplog.at_level(logging.INFO):
            with monitor.monitor_context("test_operation"):
                time.sleep(0.1)

        assert "Monitor context 'test_operation' completed" in caplog.text
        assert "System Monitor Summary" in caplog.text

    def test_monitor_context_without_tag(self, monitor, mock_psutil, caplog):
        """Test monitor context manager without tag"""
        monitor._collect_sample()

        with caplog.at_level(logging.INFO):
            with monitor.monitor_context():
                time.sleep(0.1)

        # Should still log summary but not the completion message
        assert "System Monitor Summary" in caplog.text
        assert "Monitor context" not in caplog.text


# Tests for error handling
class TestErrorHandling:
    def test_metric_collection_error(self, monitor, mock_psutil, monkeypatch, caplog):
        """Test handling of errors during metric collection"""

        def failing_collector():
            raise RuntimeError("Collection failed")

        # Add a failing collector
        monitor._metric_collectors["failing_metric"] = failing_collector
        monitor._metrics["failing_metric"] = deque(maxlen=monitor.history_size)

        with caplog.at_level(logging.WARNING):
            monitor._collect_sample()

        assert "Failed to collect metric 'failing_metric'" in caplog.text

        # Other metrics should still be collected
        latest = monitor.get_latest()
        assert latest["cpu_percent"] == 50.0

    def test_gpu_error_handling(self, monitor, mock_psutil, mock_torch_cuda, monkeypatch):
        """Test handling of GPU-related errors"""

        # Skip if CUDA is not available on this platform
        if mock_torch_cuda is None:
            pytest.skip("CUDA not available on this platform")

        # Make GPU utilization fail
        def failing_utilization(device):
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(torch.cuda, "utilization", failing_utilization)

        # Reinitialize to pick up GPU metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        # Should handle error gracefully
        latest = monitor.get_latest()
        assert latest["gpu_utilization_avg"] == 0.0
