import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import torch


class SystemMonitor:
    """A utility class for monitoring system statistics with support for multiple metrics.

    Monitors CPU, memory, GPU (if available), and process metrics with configurable
    sampling intervals and history retention. Cross-platform compatible and designed
    for both local development and containerized environments.
    """

    def __init__(
        self,
        sampling_interval_sec: float = 1.0,
        history_size: int = 100,
        logger: Optional[logging.Logger] = None,
        auto_start: bool = True,
    ):
        """Initialize the machine monitor.

        Args:
            sampling_interval_sec: Seconds between metric samples
            history_size: Number of samples to retain per metric
            logger: Optional logger instance
            auto_start: Whether to start monitoring immediately
        """
        self.logger = logger or logging.getLogger("SystemMonitor")
        self.sampling_interval_sec = sampling_interval_sec
        self.history_size = history_size

        # Thread control
        self._thread: Optional[Thread] = None
        self._stop_flag = False
        self._lock = Lock()

        # Metric storage
        self._metrics: Dict[str, deque] = {}
        self._latest: Dict[str, Any] = {}
        self._metric_collectors: Dict[str, Callable[[], Any]] = {}

        # Initialize default metrics
        self._initialize_default_metrics()

        if auto_start:
            self.start()

    def _initialize_default_metrics(self):
        """Set up all system metrics with cross-platform compatibility."""
        # Always available metrics
        self._metric_collectors = {
            # CPU metrics
            "cpu_percent": lambda: psutil.cpu_percent(interval=0),
            "cpu_count": lambda: psutil.cpu_count(),
            "cpu_count_logical": lambda: psutil.cpu_count(logical=True),
            "cpu_count_physical": lambda: psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
            # Memory metrics
            "memory_percent": lambda: psutil.virtual_memory().percent,
            "memory_available_mb": lambda: psutil.virtual_memory().available / (1024 * 1024),
            "memory_used_mb": lambda: psutil.virtual_memory().used / (1024 * 1024),
            "memory_total_mb": lambda: psutil.virtual_memory().total / (1024 * 1024),
            # Process-specific metrics
            "process_memory_mb": self._get_process_memory_mb,
            "process_cpu_percent": self._get_process_cpu_percent,
            "process_threads": self._get_process_threads,
        }

        # Platform-specific metrics
        # Only add temperature monitoring if the platform supports it
        sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
        if sensors_temperatures is not None:
            try:
                # Test if sensors_temperatures actually works
                temps = sensors_temperatures()
                if temps:  # Only add if we get actual data
                    self._metric_collectors["cpu_temperature"] = self._get_cpu_temperature
            except (AttributeError, OSError, NotImplementedError):
                # Some platforms have the method but it doesn't work
                self.logger.debug("Temperature sensors not functional on this platform")

        # Docker/container detection
        self._is_container = self._detect_container()
        if self._is_container:
            self.logger.info("Running in container environment")

        # GPU metrics - check multiple ways for compatibility
        self._has_gpu = False

        # Try PyTorch CUDA
        if torch.cuda.is_available():
            self._has_gpu = True
            self._gpu_backend = "cuda"
            self._metric_collectors.update(
                {
                    "gpu_count": lambda: torch.cuda.device_count(),
                    "gpu_utilization": self._get_gpu_utilization_cuda,
                    "gpu_memory_percent": self._get_gpu_memory_percent_cuda,
                    "gpu_memory_used_mb": self._get_gpu_memory_used_mb_cuda,
                }
            )
            self.logger.info(f"GPU monitoring enabled via CUDA ({torch.cuda.device_count()} devices)")

        # Try PyTorch MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._has_gpu = True
            self._gpu_backend = "mps"
            self._metric_collectors.update(
                {
                    "gpu_count": lambda: 1,  # MPS presents as single device
                    "gpu_available": lambda: torch.backends.mps.is_available(),
                }
            )
            self.logger.info("GPU monitoring enabled via MPS (Apple Silicon)")

        # Initialize history storage for all metrics
        for name in self._metric_collectors:
            self._metrics[name] = deque(maxlen=self.history_size)

    def _detect_container(self) -> bool:
        """Detect if running in a container (Docker, Kubernetes, etc)."""
        # Check for /.dockerenv file
        if os.path.exists("/.dockerenv"):
            return True

        # Check cgroup for docker/kubernetes
        try:
            with open("/proc/1/cgroup", "r") as f:
                if any("docker" in line or "kubepods" in line for line in f):
                    return True
        except (FileNotFoundError, PermissionError, IOError):
            pass

        # Check for container environment variables
        return any(var in os.environ for var in ["KUBERNETES_SERVICE_HOST", "DOCKER_CONTAINER", "container"])

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available (cross-platform)."""
        try:
            sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
            if sensors_temperatures is None:
                return None

            temps = sensors_temperatures()
            if not temps:
                return None

            # Try common sensor names across platforms
            for name in ["coretemp", "cpu_thermal", "cpu-thermal", "k10temp", "zenpower"]:
                if name in temps and temps[name]:
                    return temps[name][0].current

            # Fallback: return first available temperature
            for entries in temps.values():
                if entries:
                    return entries[0].current

        except (AttributeError, OSError, IOError) as e:
            # AttributeError: In case the sensor object doesn't have expected attributes
            # OSError/IOError: Common when sensors are not accessible (permissions, hardware)
            self.logger.debug(f"Failed to read CPU temperature: {type(e).__name__}: {e}")
        except Exception as e:
            # Catch any other unexpected errors and log them
            self.logger.warning(f"Unexpected error reading CPU temperature: {type(e).__name__}: {e}")

        return None

    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_process_cpu_percent(self) -> float:
        """Get current process CPU usage percent."""
        try:
            process = psutil.Process(os.getpid())
            return process.cpu_percent()
        except Exception:
            return 0.0

    def _get_process_threads(self) -> int:
        """Get current process thread count."""
        try:
            process = psutil.Process()
            return process.num_threads()
        except Exception:
            return 0

    def _get_gpu_utilization_cuda(self) -> float:
        """Get average GPU utilization across all CUDA GPUs."""
        try:
            utils = []
            for i in range(torch.cuda.device_count()):
                # Handle potential errors for specific GPUs
                try:
                    utils.append(torch.cuda.utilization(i))
                except (RuntimeError, torch.cuda.CudaError) as e:
                    # RuntimeError: Common when CUDA is not properly initialized or device is unavailable
                    # CudaError: Specific CUDA-related errors
                    self.logger.debug(f"Failed to get utilization for GPU {i}: {type(e).__name__}: {e}")
                    utils.append(0)
                except Exception as e:
                    # Unexpected errors
                    self.logger.warning(f"Unexpected error getting GPU {i} utilization: {type(e).__name__}: {e}")
                    utils.append(0)
            return sum(utils) / len(utils) if utils else 0
        except (RuntimeError, AttributeError) as e:
            # RuntimeError: CUDA not available or not initialized
            # AttributeError: torch.cuda module issues
            self.logger.debug(f"Failed to get GPU utilization: {type(e).__name__}: {e}")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU utilization: {type(e).__name__}: {e}")
            return 0.0

    def _get_gpu_memory_percent_cuda(self) -> float:
        """Get average GPU memory usage percent across all CUDA GPUs."""
        try:
            percents = []
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    percents.append((total - free) / total * 100)
                except (RuntimeError, torch.cuda.CudaError) as e:
                    # RuntimeError: Device not available or CUDA error
                    # CudaError: Specific CUDA memory query errors
                    self.logger.debug(f"Failed to get memory info for GPU {i}: {type(e).__name__}: {e}")
                    continue
                except ZeroDivisionError:
                    # In case total memory is reported as 0 (shouldn't happen but defensive programming)
                    self.logger.warning(f"GPU {i} reports 0 total memory")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error getting GPU {i} memory: {type(e).__name__}: {e}")
                    continue
            return sum(percents) / len(percents) if percents else 0
        except (RuntimeError, AttributeError) as e:
            self.logger.debug(f"Failed to get GPU memory percent: {type(e).__name__}: {e}")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU memory percent: {type(e).__name__}: {e}")
            return 0.0

    def _get_gpu_memory_used_mb_cuda(self) -> float:
        """Get total GPU memory used across all CUDA GPUs in MB."""
        try:
            total_used = 0
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    total_used += (total - free) / (1024 * 1024)
                except (RuntimeError, torch.cuda.CudaError) as e:
                    # RuntimeError: Device not available or CUDA error
                    # CudaError: Specific CUDA memory query errors
                    self.logger.debug(f"Failed to get memory info for GPU {i}: {type(e).__name__}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error getting GPU {i} memory: {type(e).__name__}: {e}")
                    continue
            return total_used
        except (RuntimeError, AttributeError) as e:
            self.logger.debug(f"Failed to get GPU memory used: {type(e).__name__}: {e}")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU memory used: {type(e).__name__}: {e}")
            return 0.0

    def _collect_sample(self):
        """Collect a single sample of all metrics."""
        timestamp = time.time()

        for name, collector in self._metric_collectors.items():
            try:
                value = collector()

                with self._lock:
                    self._metrics[name].append((timestamp, value))
                    self._latest[name] = value

            except Exception as e:
                self.logger.warning(f"Failed to collect metric '{name}': {e}")

    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        self.logger.debug("Monitor thread started")

        while not self._stop_flag:
            self._collect_sample()
            time.sleep(self.sampling_interval_sec)

        self.logger.debug("Monitor thread stopped")

    def start(self):
        """Start the monitoring thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.logger.warning("Monitor already running")
                return

            self._stop_flag = False
            self._thread = Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            self.logger.info("System monitoring started")

    def stop(self):
        """Stop the monitoring thread."""
        if not self._thread or not self._thread.is_alive():
            return

        self._stop_flag = True
        self._thread.join(timeout=self.sampling_interval_sec * 2)
        self.logger.info("System monitoring stopped")

    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._thread is not None and self._thread.is_alive()

    def get_latest(self, metric: Optional[str] = None) -> Any:
        """Get the most recent value(s).

        Args:
            metric: Specific metric name, or None for all metrics

        Returns:
            Single metric value or dict of all latest values
        """
        with self._lock:
            if metric:
                return self._latest.get(metric)
            return self._latest.copy()

    def get_history(self, metric: str) -> List[Tuple[float, Any]]:
        """Get historical values for a metric.

        Args:
            metric: Metric name

        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            if metric not in self._metrics:
                return []
            return list(self._metrics[metric])

    @contextmanager
    def monitor_context(self, tag: Optional[str] = None):
        """Context manager for monitoring a code block.

        Records metrics before and after the block execution.

        Args:
            tag: Optional tag for logging

        Usage:
            with monitor.monitor_context("heavy_computation"):
                # code to monitor
                pass
        """
        start_time = time.time()

        yield

        elapsed = time.time() - start_time

        if tag:
            self.logger.info(f"Monitor context '{tag}' completed in {elapsed:.3f}s")

        self.log_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics.

        Returns:
            Dict with current values and statistics over all history
        """
        summary = {"timestamp": time.time(), "metrics": {}}

        for metric in self._metric_collectors:
            # Get all history for this metric
            history = self.get_history(metric)

            # Calculate statistics from history
            latest = self.get_latest(metric)
            avg = None
            min_val = None
            max_val = None

            if history:
                values = [val for ts, val in history if val is not None]
                if values:
                    avg = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)

            summary["metrics"][metric] = {
                "latest": latest,
                "average": avg,
                "min": min_val,
                "max": max_val,
                "sample_count": len(history),
            }

        return summary

    def log_summary(self, level: int = logging.INFO):
        """Log a summary of current metrics.

        Args:
            level: Logging level
        """
        summary = self.get_summary()

        self.logger.log(level, f"System Monitor Summary (buffer size: {self.history_size}):")

        for metric, stats in summary["metrics"].items():
            parts = [f"{metric}:"]

            if stats["latest"] is not None:
                parts.append(f"current={stats['latest']:.1f}")
            if stats["average"] is not None:
                parts.append(f"avg={stats['average']:.1f}")
            if stats["min"] is not None and stats["max"] is not None:
                parts.append(f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")
            parts.append(f"samples={stats['sample_count']}")

            self.logger.log(level, "  " + " ".join(parts))

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics.

        Returns:
            List of metric names
        """
        return list(self._metric_collectors.keys())

    def stats(self) -> dict[str, float]:
        stats = {}
        summary = self.get_summary()
        for metric_name, metric_data in summary["metrics"].items():
            if metric_data["latest"] is not None:
                stats[f"monitor/{metric_name}"] = metric_data["latest"]
        return stats
