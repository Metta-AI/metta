import logging
import platform
import time
from collections import deque
from typing import Generator

import numpy as np
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
    if monitor.is_running():
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
        assert not monitor.is_running()
        assert len(monitor.get_available_metrics()) > 0

    def test_init_custom_params(self, mock_psutil):
        """Test initialization with custom parameters"""
        monitor = SystemMonitor(sampling_interval_sec=2.0, history_size=50, auto_start=False)

        assert monitor.sampling_interval_sec == 2.0
        assert monitor.history_size == 50

    def test_auto_start(self, mock_psutil):
        """Test auto_start parameter"""
        monitor = SystemMonitor(auto_start=True, sampling_interval_sec=0.1)

        assert monitor.is_running()
        monitor.stop()

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
        latest = monitor.get_latest()

        # Should have non-zero CPU after initialization
        assert latest["process_cpu_percent"] == 25.0

    def test_memory_metrics(self, monitor, mock_psutil):
        """Test memory metric collection"""
        monitor._collect_sample()

        latest = monitor.get_latest()
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
        """Test starting and stopping monitoring"""
        assert not monitor.is_running()

        monitor.start()
        assert monitor.is_running()

        monitor.stop()
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


# Integration tests
class TestIntegration:
    def test_full_monitoring_cycle(self, mock_psutil):
        """Test full monitoring lifecycle"""
        monitor = SystemMonitor(sampling_interval_sec=0.05, history_size=5, auto_start=True)

        # Let it collect some samples
        time.sleep(0.2)

        # Check we have data
        latest = monitor.get_latest()
        assert len(latest) > 0

        # Check history
        history = monitor.get_history("cpu_percent")
        assert len(history) >= 3

        # Get summary
        summary = monitor.get_summary()
        assert summary["metrics"]["cpu_percent"]["sample_count"] >= 3

        # Stop monitoring
        monitor.stop()
        assert not monitor.is_running()

    def test_concurrent_access(self, monitor, mock_psutil):
        """Test thread-safe concurrent access"""
        import threading

        results = {"errors": []}

        def reader_thread():
            try:
                for _ in range(10):
                    monitor.get_latest()
                    monitor.get_history("cpu_percent")
                    time.sleep(0.01)
            except Exception as e:
                results["errors"].append(e)

        # Start monitoring
        monitor.start()

        # Start multiple reader threads
        threads = [threading.Thread(target=reader_thread) for _ in range(3)]
        for t in threads:
            t.start()

        # Let them run
        time.sleep(0.2)

        # Wait for completion
        for t in threads:
            t.join()

        monitor.stop()

        # Should have no errors
        assert len(results["errors"]) == 0


class TestPlatformSpecific:
    def test_no_gpu_available(self, mock_psutil, monkeypatch):
        """Test behavior when no GPU is available"""
        # Save original state
        _original_cuda_available = torch.cuda.is_available() if hasattr(torch.cuda, "is_available") else False
        _original_mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False

        # Mock both CUDA and MPS as unavailable
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # Create a fresh monitor instance with GPU support mocked as unavailable
        monitor = SystemMonitor(sampling_interval_sec=0.1, history_size=10, auto_start=False)

        # GPU metrics should not be available
        metrics = monitor.get_available_metrics()
        assert "gpu_count" not in metrics
        assert "gpu_utilization_avg" not in metrics
        assert "gpu_memory_percent_avg" not in metrics
        assert "gpu_memory_used_mb_total" not in metrics
        assert "gpu_available" not in metrics
        # Also check per-GPU metrics aren't created
        assert not any(metric.startswith("gpu0_") for metric in metrics)
        assert not any(metric.startswith("gpu1_") for metric in metrics)


class TestRealSystemMonitoring:
    """Tests that use the real system (not mocked)"""

    @pytest.mark.slow
    def test_real_monitoring_memory_pattern(self):
        """Test monitoring memory allocation and deallocation patterns"""
        monitor = SystemMonitor(
            sampling_interval_sec=0.05,
            history_size=200,
            auto_start=True,
        )

        # Baseline
        time.sleep(0.2)
        baseline_process_memory = monitor.get_latest("process_memory_mb")

        # Memory allocation pattern
        arrays = []
        allocation_sizes = [100, 200, 400, 800]  # MB

        for size_mb in allocation_sizes:
            # Allocate memory
            size_elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
            arr = np.random.rand(size_elements)
            arrays.append(arr)

            # Perform operation to ensure memory is actually allocated
            _ = arr.sum()

            # Wait for monitor to capture
            time.sleep(0.2)

        # Peak memory usage
        peak_with_arrays = monitor.get_latest("process_memory_mb")

        # Deallocate
        arrays.clear()
        del arrays

        # Force garbage collection
        import gc

        gc.collect()

        # Platform-specific memory release behavior
        if IS_WINDOWS:
            # Windows tends to hold onto memory longer
            time.sleep(2.0)
            _threshold = 500
        elif IS_MACOS:
            # macOS has different memory management
            time.sleep(1.5)
            _threshold = 300
        else:
            # Linux
            time.sleep(1.0)
            _threshold = 200

        # Final memory
        final_memory = monitor.get_latest("process_memory_mb")

        monitor.stop()

        # Analyze memory pattern
        memory_history = monitor.get_history("process_memory_mb")
        _memory_values = [value for _, value in memory_history]

        # Verify we captured the memory spike (more lenient for different platforms)
        expected_increase = sum(allocation_sizes) * (0.3 if IS_MACOS else 0.5)
        assert peak_with_arrays > baseline_process_memory + expected_increase, (
            f"Memory didn't increase as expected: baseline={baseline_process_memory:.1f}MB, "
            f"peak={peak_with_arrays:.1f}MB, expected increase={expected_increase:.1f}MB"
        )

        # Platform-aware memory release check
        if IS_MACOS:
            # macOS may not release memory as aggressively
            # Just check that we're not still at absolute peak
            assert final_memory < peak_with_arrays, (
                f"Memory stayed at peak: peak={peak_with_arrays:.1f}MB, final={final_memory:.1f}MB"
            )
        else:
            # Other platforms should show more memory release
            assert final_memory < peak_with_arrays - 100, (
                f"Memory didn't decrease from peak: peak={peak_with_arrays:.1f}MB, final={final_memory:.1f}MB"
            )

        print(f"\nMemory Pattern Results ({platform.system()}):")
        print(f"Baseline: {baseline_process_memory:.1f}MB")
        print(f"Peak: {peak_with_arrays:.1f}MB")
        print(f"Final: {final_memory:.1f}MB")
        print(f"Memory released: {peak_with_arrays - final_memory:.1f}MB")
        print(f"Samples: {len(memory_history)}")

    @pytest.mark.slow
    def test_real_monitoring_with_context_manager(self):
        """Test real monitoring using context manager during numpy operations"""
        monitor = SystemMonitor(sampling_interval_sec=0.05, history_size=100, auto_start=True)

        # Platform-specific stabilization time
        stabilization_time = 0.5 if not IS_WINDOWS else 1.0
        time.sleep(stabilization_time)

        results = {}

        # Monitor different types of operations
        with monitor.monitor_context("numpy_linear_algebra"):
            size = 2000
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)

            # Various linear algebra operations
            _C = np.dot(A, B)
            _eigenvalues = np.linalg.eigvals(A[:500, :500])
            _inv = np.linalg.inv(A[:100, :100])

            # Capture both system and process CPU
            results["linear_algebra"] = {
                "cpu": monitor.get_latest("cpu_percent"),
                "process_cpu": monitor.get_latest("process_cpu_percent"),
            }

        time.sleep(0.5)

        with monitor.monitor_context("numpy_signal_processing"):
            signal = np.random.rand(1000000)
            _fft = np.fft.fft(signal)
            _conv = np.convolve(signal[:10000], signal[:1000], mode="full")

            results["signal_processing"] = {
                "cpu": monitor.get_latest("cpu_percent"),
                "process_cpu": monitor.get_latest("process_cpu_percent"),
            }

        time.sleep(0.5)

        with monitor.monitor_context("numpy_statistics"):
            data = np.random.randn(5000, 5000)
            _mean = np.mean(data, axis=0)
            _std = np.std(data, axis=0)
            _cov = np.cov(data[:1000, :100].T)

            results["statistics"] = {
                "cpu": monitor.get_latest("cpu_percent"),
                "process_cpu": monitor.get_latest("process_cpu_percent"),
            }

        monitor.stop()

        # Get history for both metrics
        cpu_history = monitor.get_history("cpu_percent")
        process_cpu_history = monitor.get_history("process_cpu_percent")

        max_cpu = max(value for _, value in cpu_history) if cpu_history else 0
        max_process_cpu = max(value for _, value in process_cpu_history) if process_cpu_history else 0

        # Platform-specific validation
        if IS_MACOS:
            # macOS might show different CPU patterns
            # Check either system or process CPU showed activity
            has_activity = (
                any(r["cpu"] > 0 or r["process_cpu"] > 0 for r in results.values())
                or max_cpu > 10
                or max_process_cpu > 10
            )
        else:
            # Other platforms should show clearer CPU usage
            has_activity = any(r["cpu"] > 0 for r in results.values()) or max_cpu > 20

        assert has_activity, (
            f"No CPU activity detected on {platform.system()}. "
            f"Results: {results}, Max CPU: {max_cpu:.1f}%, Max Process CPU: {max_process_cpu:.1f}%"
        )

        print(f"\nContext Manager Monitoring Results ({platform.system()}):")
        for operation, metrics in results.items():
            print(f"{operation}: System CPU={metrics['cpu']:.1f}%, Process CPU={metrics['process_cpu']:.1f}%")
        print(f"Max System CPU: {max_cpu:.1f}%")
        print(f"Max Process CPU: {max_process_cpu:.1f}%")


if __name__ == "__main__":
    """Run a comprehensive integration test of SystemMonitor when executed directly."""
    import sys

    print("=" * 80)
    print("SystemMonitor Integration Test")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {HAS_CUDA}")
    print(f"MPS Available: {HAS_MPS}")
    print("=" * 80)

    # Create monitor with faster sampling for testing
    print("\nInitializing SystemMonitor...")
    system_monitor = SystemMonitor(sampling_interval_sec=0.5, history_size=20)

    # Let it collect some samples
    print("Collecting samples for 3 seconds...")
    time.sleep(3)

    # Display current stats
    stats = system_monitor.stats()
    print("\nCurrent System Stats:")
    print("-" * 40)

    # Group metrics by category for better readability
    cpu_metrics = {}
    memory_metrics = {}
    process_metrics = {}
    gpu_aggregate_metrics = {}
    gpu_individual_metrics = {}
    other_metrics = {}

    for key, value in sorted(stats.items()):
        metric_name = key.replace("monitor/", "")
        if metric_name.startswith("cpu_"):
            cpu_metrics[metric_name] = value
        elif metric_name.startswith("memory_"):
            memory_metrics[metric_name] = value
        elif metric_name.startswith("process_"):
            process_metrics[metric_name] = value
        elif metric_name.startswith("gpu") and "_" in metric_name[3:]:
            # Individual GPU metrics like gpu0_utilization
            gpu_individual_metrics[metric_name] = value
        elif metric_name.startswith("gpu_"):
            # Aggregate GPU metrics
            gpu_aggregate_metrics[metric_name] = value
        else:
            other_metrics[metric_name] = value

    # Display CPU metrics
    print("\nCPU Metrics:")
    for metric, value in cpu_metrics.items():
        if metric == "cpu_temperature" and value == -273.15:
            print(f"  {metric}: Not available")
        else:
            print(f"  {metric}: {value:.2f}")

    # Display Memory metrics
    print("\nMemory Metrics:")
    for metric, value in memory_metrics.items():
        print(f"  {metric}: {value:.2f}")

    # Display Process metrics
    print("\nProcess Metrics:")
    for metric, value in process_metrics.items():
        print(f"  {metric}: {value:.2f}")

    # Check for specific issues
    print("\nDiagnostics:")
    print("-" * 40)
    process_cpu = stats.get("monitor/process_cpu_percent", None)
    if process_cpu is None:
        print("❌ Process CPU: Not found in stats")
    elif process_cpu == 0:
        print("⚠️  Process CPU: Still zero (may need more time to initialize)")
    else:
        print(f"✅ Process CPU: {process_cpu:.2f}% (working correctly)")

    cpu_temp = stats.get("monitor/cpu_temperature", None)
    if cpu_temp is None:
        print("✅ CPU Temperature: Not available (correctly excluded)")
    elif cpu_temp == -273.15:
        print("❌ CPU Temperature: Shows -273.15 (should be excluded)")
    else:
        print(f"✅ CPU Temperature: {cpu_temp:.1f}°C")

    # Display GPU metrics if available
    if gpu_aggregate_metrics or gpu_individual_metrics:
        print("\nGPU Metrics:")
        print("-" * 40)

        # Show aggregate metrics
        if gpu_aggregate_metrics:
            print("Aggregate Metrics:")
            for metric, value in sorted(gpu_aggregate_metrics.items()):
                print(f"  {metric}: {value:.2f}")

        # Show per-GPU metrics
        if gpu_individual_metrics:
            print("\nPer-GPU Metrics:")
            # Organize by GPU index
            gpu_data = {}
            for metric, value in gpu_individual_metrics.items():
                # Extract GPU index (e.g., gpu0_utilization -> 0)
                parts = metric.split("_", 1)
                if parts[0].startswith("gpu"):
                    gpu_idx = parts[0][3:]  # Remove 'gpu' prefix
                    if gpu_idx not in gpu_data:
                        gpu_data[gpu_idx] = {}
                    gpu_data[gpu_idx][parts[1]] = value

            for gpu_idx in sorted(gpu_data.keys()):
                print(f"  GPU {gpu_idx}:")
                for metric, value in sorted(gpu_data[gpu_idx].items()):
                    print(f"    {metric}: {value:.2f}")
    else:
        print("\nNo GPU metrics available")

    # Show history for a metric
    print("\nSample History (CPU Percent):")
    print("-" * 40)
    cpu_history = system_monitor.get_history("cpu_percent")
    if cpu_history:
        # Show last 5 samples
        for timestamp, value in cpu_history[-5:]:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(timestamp))}: {value:.2f}%")

    # Test context manager with some CPU work
    print("\nTesting Context Manager with CPU workload...")
    print("-" * 40)

    with system_monitor.monitor_context("matrix_multiplication"):
        # Do some CPU-intensive work
        size = 1000
        import numpy as np

        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        C = np.dot(A, B)
        print(f"Computed {size}x{size} matrix multiplication")

    # Get summary
    print("\nMetric Summary:")
    print("-" * 40)
    summary = system_monitor.get_summary()

    # Show summary for key metrics
    key_metrics = ["cpu_percent", "memory_percent", "process_cpu_percent", "process_memory_mb"]
    if "gpu_utilization_avg" in summary["metrics"]:
        key_metrics.extend(["gpu_utilization_avg", "gpu_memory_percent_avg"])

    for metric in key_metrics:
        if metric in summary["metrics"]:
            stats = summary["metrics"][metric]
            if stats["latest"] is not None:
                print(f"{metric}:")
                print(f"  Latest: {stats['latest']:.2f}")
                if stats["average"] is not None:
                    print(f"  Average: {stats['average']:.2f}")
                if stats["min"] is not None and stats["max"] is not None:
                    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

    # Stop monitoring
    system_monitor.stop()
    print("\n✅ Integration test completed successfully!")
    print("=" * 80)
