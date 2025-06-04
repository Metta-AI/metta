import logging
import time
from collections import deque

import numpy as np
import psutil
import pytest
import torch

from metta.util.system_monitor import SystemMonitor


# Test doubles and helpers
class FakeProcess:
    """Simple test double for psutil.Process"""

    def __init__(self, memory_rss=100 * 1024 * 1024, cpu_percent=25.0, num_threads=4):
        self._memory_rss = memory_rss
        self._cpu_percent = cpu_percent
        self._num_threads = num_threads

    def memory_info(self):
        return type("MemInfo", (), {"rss": self._memory_rss})()

    def cpu_percent(self):
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
def monitor():
    """Create a SystemMonitor instance with auto_start=False"""
    monitor = SystemMonitor(sampling_interval_sec=0.1, history_size=10, auto_start=False)
    yield monitor
    # Cleanup
    if monitor.is_running():
        monitor.stop()


@pytest.fixture
def mock_psutil(monkeypatch):
    """Mock psutil functions with controllable values"""
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
        return mock_data["process"]

    def mock_sensors_temperatures():
        return mock_data["temperatures"]

    monkeypatch.setattr(psutil, "cpu_percent", mock_cpu_percent)
    monkeypatch.setattr(psutil, "cpu_count", mock_cpu_count)
    monkeypatch.setattr(psutil, "virtual_memory", mock_virtual_memory)
    monkeypatch.setattr(psutil, "Process", mock_process)

    if hasattr(psutil, "sensors_temperatures"):
        monkeypatch.setattr(psutil, "sensors_temperatures", mock_sensors_temperatures)

    return mock_data


@pytest.fixture
def mock_torch_cuda(monkeypatch):
    """Mock torch CUDA functions"""
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


# Tests for metric collection
class TestMetricCollection:
    def test_cpu_metrics(self, monitor, mock_psutil):
        """Test CPU metric collection"""
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert latest["cpu_percent"] == 50.0
        assert latest["cpu_count"] == 8
        assert latest["cpu_count_logical"] == 8
        assert latest["cpu_count_physical"] == 4

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
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert latest["process_memory_mb"] == 100.0
        assert latest["process_cpu_percent"] == 25.0
        assert latest["process_threads"] == 4

    def test_gpu_metrics_cuda(self, monitor, mock_psutil, mock_torch_cuda):
        """Test GPU metric collection with CUDA"""
        # Reinitialize to pick up GPU metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert latest["gpu_count"] == 2
        assert latest["gpu_utilization"] == 50.0  # Average of 60 and 40
        assert latest["gpu_memory_percent"] == 68.75  # Average of 75% and 62.5%
        assert latest["gpu_memory_used_mb"] == 11264.0  # Total used across GPUs

    def test_temperature_metrics(self, monitor, mock_psutil):
        """Test CPU temperature collection"""
        # Add temperature data
        mock_psutil["temperatures"] = {"coretemp": [type("Temp", (), {"current": 65.0})()]}

        # Reinitialize to pick up temperature metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        latest = monitor.get_latest()
        assert latest.get("cpu_temperature") == 65.0


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
        for i in range(15):
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

        # Make GPU utilization fail
        def failing_utilization(device):
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(torch.cuda, "utilization", failing_utilization)

        # Reinitialize to pick up GPU metrics
        monitor._initialize_default_metrics()
        monitor._collect_sample()

        # Should handle error gracefully
        latest = monitor.get_latest()
        assert latest["gpu_utilization"] == 0.0


# Tests for platform-specific behavior
class TestPlatformSpecific:
    def test_container_detection(self, monitor, monkeypatch, tmp_path):
        """Test container environment detection"""
        # Test with /.dockerenv file
        dockerenv = tmp_path / ".dockerenv"
        dockerenv.touch()
        monkeypatch.setattr("os.path.exists", lambda path: path == "/.dockerenv")

        assert monitor._detect_container() is True

    def test_no_gpu_available(self, monitor, mock_psutil, monkeypatch):
        """Test behavior when no GPU is available"""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Reinitialize
        monitor._initialize_default_metrics()

        # GPU metrics should not be available
        metrics = monitor.get_available_metrics()
        assert "gpu_count" not in metrics
        assert "gpu_utilization" not in metrics


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


# Real system tests (not mocked)
class TestRealSystemMonitoring:
    """Tests that use real psutil to monitor actual system load"""

    @pytest.mark.slow
    def test_real_monitoring_with_numpy_load(self):
        """Test monitoring real system metrics during heavy numpy computations"""
        # Create monitor with real psutil (no mocking)
        monitor = SystemMonitor(
            sampling_interval_sec=0.1,  # Sample every 100ms
            history_size=100,
            auto_start=True,
        )

        # Ensure we have baseline measurements
        time.sleep(0.3)

        # Get baseline metrics
        baseline_cpu = monitor.get_latest("cpu_percent")
        _baseline_memory = monitor.get_latest("memory_used_mb")
        baseline_process_memory = monitor.get_latest("process_memory_mb")

        # Record start time
        start_time = time.time()

        # Perform heavy numpy computations
        results = []
        matrix_sizes = [1000, 2000, 3000]

        for size in matrix_sizes:
            # Matrix multiplication - CPU intensive
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            C = np.dot(A, B)
            results.append(C.sum())

            # SVD decomposition - CPU and memory intensive
            U, S, Vt = np.linalg.svd(A)
            results.append(S.sum())

            # Large array operations - memory intensive
            large_array = np.random.rand(size * size)
            sorted_array = np.sort(large_array)
            results.append(sorted_array[0])

            # FFT - CPU intensive
            fft_result = np.fft.fft2(A)
            results.append(np.abs(fft_result).sum())

            # Give monitor time to sample during computation
            time.sleep(0.1)

        # Let monitoring catch up
        time.sleep(0.5)
        computation_time = time.time() - start_time

        # Stop monitoring
        monitor.stop()

        # Get summary of what happened
        summary = monitor.get_summary()

        # Verify monitoring captured the load
        cpu_stats = summary["metrics"]["cpu_percent"]
        _memory_stats = summary["metrics"]["memory_used_mb"]
        process_memory_stats = summary["metrics"]["process_memory_mb"]

        # CPU should have peaked during computation
        assert cpu_stats["max"] > baseline_cpu, f"CPU didn't increase: baseline={baseline_cpu}, max={cpu_stats['max']}"

        # Process memory should have increased
        assert process_memory_stats["max"] > baseline_process_memory * 1.1, (
            f"Process memory didn't increase significantly: baseline={baseline_process_memory}, max={process_memory_stats['max']}"
        )

        # We should have collected sufficient samples
        assert cpu_stats["sample_count"] >= int(computation_time / 0.1 * 0.8), (
            f"Not enough samples collected: {cpu_stats['sample_count']}"
        )

        # Verify computation actually ran (sanity check)
        assert len(results) == len(matrix_sizes) * 4
        assert all(isinstance(r, (float, np.float64)) for r in results)

        # Log interesting metrics
        print("\nReal System Monitoring Results:")
        print(f"Computation time: {computation_time:.2f}s")
        print(f"CPU: baseline={baseline_cpu:.1f}%, max={cpu_stats['max']:.1f}%, avg={cpu_stats['average']:.1f}%")
        print(f"Process Memory: baseline={baseline_process_memory:.1f}MB, max={process_memory_stats['max']:.1f}MB")
        print(f"Samples collected: {cpu_stats['sample_count']}")

    @pytest.mark.slow
    def test_real_monitoring_memory_pattern(self):
        """Test monitoring memory allocation and deallocation patterns"""
        monitor = SystemMonitor(
            sampling_interval_sec=0.05,  # Sample every 50ms for finer granularity
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

        # Force garbage collection
        import gc

        gc.collect()

        # Wait for memory to be released
        time.sleep(0.5)

        # Final memory
        final_memory = monitor.get_latest("process_memory_mb")

        monitor.stop()

        # Analyze memory pattern
        memory_history = monitor.get_history("process_memory_mb")
        memory_values = [value for _, value in memory_history]

        # Verify we captured the memory spike
        assert peak_with_arrays > baseline_process_memory + sum(allocation_sizes) * 0.8, (
            f"Memory didn't increase as expected: baseline={baseline_process_memory:.1f}MB, peak={peak_with_arrays:.1f}MB"
        )

        # Verify memory was released (allowing for some overhead)
        assert final_memory < baseline_process_memory + 100, (
            f"Memory wasn't released: final={final_memory:.1f}MB, baseline={baseline_process_memory:.1f}MB"
        )

        # Verify we captured the pattern
        assert max(memory_values) > min(memory_values) + sum(allocation_sizes) * 0.8

        print("\nMemory Pattern Results:")
        print(f"Baseline: {baseline_process_memory:.1f}MB")
        print(f"Peak: {peak_with_arrays:.1f}MB (expected ~{baseline_process_memory + sum(allocation_sizes):.1f}MB)")
        print(f"Final: {final_memory:.1f}MB")
        print(f"Samples: {len(memory_history)}")

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_real_gpu_monitoring(self):
        """Test real GPU monitoring during CUDA operations"""
        monitor = SystemMonitor(sampling_interval_sec=0.1, history_size=100, auto_start=True)

        # Baseline
        time.sleep(0.3)
        baseline_gpu_util = monitor.get_latest("gpu_utilization")
        baseline_gpu_memory = monitor.get_latest("gpu_memory_used_mb")

        # GPU operations
        device = torch.device("cuda")
        tensors = []

        # Perform GPU-intensive operations
        for size in [1000, 2000, 4000]:
            # Create large tensors on GPU
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)

            # Matrix multiplication
            C = torch.matmul(A, B)
            tensors.append(C)

            # More operations
            D = torch.svd(A).U
            E = torch.fft.fft2(A.to(torch.complex64))

            tensors.extend([D, E])

            # Ensure operations complete
            torch.cuda.synchronize()
            time.sleep(0.2)

        # Peak usage
        time.sleep(0.3)

        # Cleanup
        tensors.clear()
        torch.cuda.empty_cache()

        time.sleep(0.5)
        monitor.stop()

        # Analyze GPU metrics
        summary = monitor.get_summary()
        gpu_util_stats = summary["metrics"]["gpu_utilization"]
        gpu_memory_stats = summary["metrics"]["gpu_memory_used_mb"]

        # Verify GPU was utilized
        assert gpu_util_stats["max"] > baseline_gpu_util + 10, (
            f"GPU utilization didn't increase: baseline={baseline_gpu_util}%, max={gpu_util_stats['max']}%"
        )

        assert gpu_memory_stats["max"] > baseline_gpu_memory + 100, (
            f"GPU memory didn't increase: baseline={baseline_gpu_memory}MB, max={gpu_memory_stats['max']}MB"
        )

        print("\nGPU Monitoring Results:")
        print(f"GPU Utilization: max={gpu_util_stats['max']:.1f}%, avg={gpu_util_stats['average']:.1f}%")
        print(f"GPU Memory: max={gpu_memory_stats['max']:.1f}MB, avg={gpu_memory_stats['average']:.1f}MB")

    @pytest.mark.slow
    def test_real_monitoring_with_context_manager(self):
        """Test real monitoring using context manager during numpy operations"""
        monitor = SystemMonitor(sampling_interval_sec=0.05, history_size=100, auto_start=True)

        # Let it stabilize
        time.sleep(0.2)

        results = {}

        # Monitor different types of operations
        with monitor.monitor_context("numpy_linear_algebra"):
            size = 2000
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)

            # Various linear algebra operations
            _C = np.dot(A, B)
            _eigenvalues = np.linalg.eigvals(A[:500, :500])  # Smaller for speed
            _inv = np.linalg.inv(A[:100, :100])  # Even smaller

            results["linear_algebra"] = monitor.get_latest("cpu_percent")

        # Let system settle
        time.sleep(0.5)

        with monitor.monitor_context("numpy_signal_processing"):
            # Signal processing operations
            signal = np.random.rand(1000000)
            _fft = np.fft.fft(signal)
            _conv = np.convolve(signal[:10000], signal[:1000], mode="full")

            results["signal_processing"] = monitor.get_latest("cpu_percent")

        # Let system settle
        time.sleep(0.5)

        with monitor.monitor_context("numpy_statistics"):
            # Statistical operations
            data = np.random.randn(5000, 5000)
            _mean = np.mean(data, axis=0)
            _std = np.std(data, axis=0)
            _cov = np.cov(data[:1000, :100].T)  # Smaller subset

            results["statistics"] = monitor.get_latest("cpu_percent")

        monitor.stop()

        # Verify we captured different workload patterns
        assert results["linear_algebra"] > 0
        assert results["signal_processing"] > 0
        assert results["statistics"] > 0

        # These operations should have caused noticeable CPU usage
        assert any(cpu > 20 for cpu in results.values()), f"No significant CPU usage detected: {results}"

        print("\nContext Manager Monitoring Results:")
        for operation, cpu in results.items():
            print(f"{operation}: {cpu:.1f}% CPU")
