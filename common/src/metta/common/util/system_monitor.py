import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from threading import Lock, Thread
from typing import Any, Callable, Tuple

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
        logger: logging.Logger | None = None,
        auto_start: bool = True,
        external_timer: Any | None = None,
    ):
        """Initialize the machine monitor.

        Args:
            sampling_interval_sec: Seconds between metric samples
            history_size: Number of samples to retain per metric
            logger: Optional logger instance
            auto_start: Whether to start monitoring immediately
            external_timer: Optional external timer (e.g., trainer's Stopwatch) for elapsed time
        """
        self.logger = logger or logging.getLogger("SystemMonitor")
        self.sampling_interval_sec = sampling_interval_sec
        self.history_size = history_size

        # Initialize process object for CPU tracking
        self._process = psutil.Process(os.getpid())
        # Call cpu_percent once to initialize the baseline
        self._process.cpu_percent()

        # Thread control
        self._thread: Thread | None = None
        self._stop_flag = False
        self._lock = Lock()
        self._start_time: float | None = None  # Track when monitoring started
        self._external_timer = external_timer  # External timer for elapsed time

        # Metric storage
        self._metrics: dict[str, deque] = {}
        self._latest: dict[str, Any] = {}
        self._metric_collectors: dict[str, Callable[[], Any]] = {}

        # Initialize default metrics
        self._initialize_default_metrics()

        if auto_start:
            self.start()

    def _initialize_default_metrics(self):
        """Set up all system metrics with cross-platform compatibility."""
        # Always available metrics
        self._metric_collectors = {
            # CPU metrics
            "cpu_percent": lambda: self._safe_get_cpu_percent(),
            "cpu_count": lambda: self._safe_get_cpu_count(),
            "cpu_count_logical": lambda: self._safe_get_cpu_count_logical(),
            "cpu_count_physical": lambda: self._safe_get_cpu_count_physical(),
            # Memory metrics
            "memory_percent": lambda: self._safe_get_memory_percent(),
            "memory_available_mb": lambda: self._safe_get_memory_available_mb(),
            "memory_used_mb": lambda: self._safe_get_memory_used_mb(),
            "memory_total_mb": lambda: self._safe_get_memory_total_mb(),
            # Process-specific metrics
            "process_memory_mb": self._get_process_memory_mb,
            "process_cpu_percent": lambda: self._safe_get_process_cpu_percent(),
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

        # Check for cost env var
        hourly_cost_str = os.environ.get("METTA_HOURLY_COST")
        if hourly_cost_str:
            try:
                total_hourly_cost = float(hourly_cost_str)
                self._metric_collectors["cost/hourly_total"] = lambda: total_hourly_cost
                # Add accrued cost metric
                self._metric_collectors["cost/accrued_total"] = lambda: self._calculate_accrued_cost(total_hourly_cost)
                self.logger.info(f"Cost monitoring enabled: ${total_hourly_cost:.4f}/hr (total for all nodes)")
            except (ValueError, TypeError):
                self.logger.warning(f"Could not parse METTA_HOURLY_COST: {hourly_cost_str}")

        # GPU metrics - check multiple ways for compatibility
        self._has_gpu = False

        # Try PyTorch CUDA
        if torch.cuda.is_available():
            self._has_gpu = True
            self._gpu_backend = "cuda"
            gpu_count = torch.cuda.device_count()

            # Add aggregate metrics (rename to make it clear they're aggregates)
            self._metric_collectors.update(
                {
                    "gpu_count": lambda: gpu_count,
                    "gpu_utilization_avg": self._get_gpu_utilization_cuda,
                    "gpu_memory_percent_avg": self._get_gpu_memory_percent_cuda,
                    "gpu_memory_used_mb_total": self._get_gpu_memory_used_mb_cuda,
                }
            )

            # Add per-GPU metrics
            for i in range(gpu_count):
                self._metric_collectors.update(
                    {
                        f"gpu{i}_utilization": lambda idx=i: self._get_single_gpu_utilization(idx),
                        f"gpu{i}_memory_percent": lambda idx=i: self._get_single_gpu_memory_percent(idx),
                        f"gpu{i}_memory_used_mb": lambda idx=i: self._get_single_gpu_memory_used_mb(idx),
                    }
                )

            self.logger.info(f"GPU monitoring enabled via CUDA ({gpu_count} devices)")

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
        # Note: /proc/1/cgroup only exists on Linux systems
        try:
            # We have a blanket catch below, so this isn't strictly necessary; but it makes things
            # less disruptive if you're debugging and running with "break on all raised exceptions".
            if os.path.exists("/proc/1/cgroup"):
                with open("/proc/1/cgroup", "r") as f:
                    if any("docker" in line or "kubepods" in line for line in f):
                        return True
        except Exception:
            # Catch all exceptions to ensure this doesn't crash on any platform
            # This includes PermissionError (no permission) and any other OS-specific errors
            pass

        # Check for container environment variables
        return any(var in os.environ for var in ["KUBERNETES_SERVICE_HOST", "DOCKER_CONTAINER", "container"])

    # Safe wrapper methods for CPU metrics
    def _safe_get_cpu_percent(self) -> float | None:
        """Get CPU percent safely."""
        try:
            return psutil.cpu_percent(interval=0)
        except Exception as e:
            self.logger.debug(f"Failed to get CPU percent: {e}")
            return None

    def _safe_get_cpu_count(self) -> int | None:
        """Get CPU count safely."""
        try:
            return psutil.cpu_count()
        except Exception as e:
            self.logger.debug(f"Failed to get CPU count: {e}")
            return None

    def _safe_get_cpu_count_logical(self) -> int | None:
        """Get logical CPU count safely."""
        try:
            return psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.debug(f"Failed to get logical CPU count: {e}")
            return None

    def _safe_get_cpu_count_physical(self) -> int | None:
        """Get physical CPU count safely."""
        try:
            return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.debug(f"Failed to get physical CPU count: {e}")
            return None

    # Safe wrapper methods for memory metrics
    def _safe_get_memory_percent(self) -> float | None:
        """Get memory percent safely."""
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            self.logger.debug(f"Failed to get memory percent: {e}")
            return None

    def _safe_get_memory_available_mb(self) -> float | None:
        """Get available memory in MB safely."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Failed to get available memory: {e}")
            return None

    def _safe_get_memory_used_mb(self) -> float | None:
        """Get used memory in MB safely."""
        try:
            return psutil.virtual_memory().used / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Failed to get used memory: {e}")
            return None

    def _safe_get_memory_total_mb(self) -> float | None:
        """Get total memory in MB safely."""
        try:
            return psutil.virtual_memory().total / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Failed to get total memory: {e}")
            return None

    def _safe_get_process_cpu_percent(self) -> float | None:
        """Get process CPU percent safely."""
        try:
            return self._process.cpu_percent()
        except Exception as e:
            self.logger.debug(f"Failed to get process CPU percent: {e}")
            return None

    def _get_cpu_temperature(self) -> float | None:
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
                    temp = temps[name][0].current
                    # Validate temperature is in reasonable range
                    if temp is not None and 20 <= temp <= 120:
                        return temp
                    elif temp is not None:
                        self.logger.debug(f"Ignoring invalid temperature {temp}°C from {name}")

            # Fallback: return first available temperature that's valid
            for sensor_name, entries in temps.items():
                if entries:
                    temp = entries[0].current
                    if temp is not None and 20 <= temp <= 120:
                        return temp
                    elif temp is not None:
                        self.logger.debug(f"Ignoring invalid temperature {temp}°C from {sensor_name}")

        except (AttributeError, OSError, IOError) as e:
            # AttributeError: In case the sensor object doesn't have expected attributes
            # OSError/IOError: Common when sensors are not accessible (permissions, hardware)
            self.logger.debug(f"Failed to read CPU temperature: {type(e).__name__}: {e}")
        except Exception as e:
            # Catch any other unexpected errors and log them
            self.logger.warning(f"Unexpected error reading CPU temperature: {type(e).__name__}: {e}")

        return None

    def _get_process_memory_mb(self) -> float | None:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Failed to get process memory: {e}")
            return None

    def _get_process_threads(self) -> int | None:
        """Get current process thread count."""
        try:
            process = psutil.Process()
            return process.num_threads()
        except Exception as e:
            self.logger.debug(f"Failed to get process threads: {e}")
            return None

    def _get_gpu_utilization_cuda(self) -> float | None:
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
            return sum(utils) / len(utils) if utils else None
        except (RuntimeError, AttributeError) as e:
            # RuntimeError: CUDA not available or not initialized
            # AttributeError: torch.cuda module issues
            self.logger.debug(f"Failed to get GPU utilization: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU utilization: {type(e).__name__}: {e}")
            return None

    def _get_gpu_memory_percent_cuda(self) -> float | None:
        """Get average GPU memory usage percent across all CUDA GPUs."""
        try:
            percents = []
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    if total > 0:  # Defensive check
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
            return sum(percents) / len(percents) if percents else None
        except (RuntimeError, AttributeError) as e:
            self.logger.debug(f"Failed to get GPU memory percent: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU memory percent: {type(e).__name__}: {e}")
            return None

    def _get_gpu_memory_used_mb_cuda(self) -> float | None:
        """Get total GPU memory used across all CUDA GPUs in MB."""
        try:
            total_used = 0
            count = 0
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    total_used += (total - free) / (1024 * 1024)
                    count += 1
                except (RuntimeError, torch.cuda.CudaError) as e:
                    # RuntimeError: Device not available or CUDA error
                    # CudaError: Specific CUDA memory query errors
                    self.logger.debug(f"Failed to get memory info for GPU {i}: {type(e).__name__}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error getting GPU {i} memory: {type(e).__name__}: {e}")
                    continue
            return total_used if count > 0 else None
        except (RuntimeError, AttributeError) as e:
            self.logger.debug(f"Failed to get GPU memory used: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU memory used: {type(e).__name__}: {e}")
            return None

    def _get_single_gpu_utilization(self, gpu_idx: int) -> float | None:
        """Get utilization for a specific GPU."""
        try:
            return torch.cuda.utilization(gpu_idx)
        except Exception as e:
            self.logger.debug(f"Failed to get utilization for GPU {gpu_idx}: {e}")
            return None

    def _get_single_gpu_memory_percent(self, gpu_idx: int) -> float | None:
        """Get memory usage percent for a specific GPU."""
        try:
            free, total = torch.cuda.mem_get_info(gpu_idx)
            if total > 0:
                return (total - free) / total * 100
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get memory percent for GPU {gpu_idx}: {e}")
            return None

    def _get_single_gpu_memory_used_mb(self, gpu_idx: int) -> float | None:
        """Get memory used in MB for a specific GPU."""
        try:
            free, total = torch.cuda.mem_get_info(gpu_idx)
            return (total - free) / (1024 * 1024)
        except Exception as e:
            self.logger.debug(f"Failed to get memory used for GPU {gpu_idx}: {e}")
            return None

    def _collect_sample(self) -> None:
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

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in separate thread)."""
        self.logger.debug("Monitor thread started")

        while not self._stop_flag:
            self._collect_sample()
            time.sleep(self.sampling_interval_sec)

        self.logger.debug("Monitor thread stopped")

    def start(self) -> None:
        """Start the monitoring thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.logger.warning("Monitor already running")
                return

            self._stop_flag = False
            if self._start_time is None:  # Only set on first start
                self._start_time = time.time()
            self._thread = Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            self.logger.info("System monitoring started")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self._thread or not self._thread.is_alive():
            return

        self._stop_flag = True
        self._thread.join(timeout=self.sampling_interval_sec * 2)
        self.logger.info("System monitoring stopped")

    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._thread is not None and self._thread.is_alive()

    def get_latest(self, metric: str | None = None) -> Any:
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

    def get_history(self, metric: str) -> list[Tuple[float, Any]]:
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
    def monitor_context(self, tag: str | None = None):
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

    def get_summary(self) -> dict[str, Any]:
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

    def log_summary(self, level: int = logging.INFO) -> None:
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

    def get_available_metrics(self) -> list[str]:
        """Get list of all available metrics.

        Returns:
            List of metric names
        """
        return list(self._metric_collectors.keys())

    def stats(self) -> dict[str, float]:
        """Get current stats, skipping any None values.

        Returns:
            Dict of metric names to float values, excluding any None values
        """
        stats = {}
        summary = self.get_summary()
        for metric_name, metric_data in summary["metrics"].items():
            if metric_data["latest"] is not None:
                stats[f"monitor/{metric_name}"] = metric_data["latest"]
        return stats

    def _calculate_accrued_cost(self, hourly_cost: float) -> float | None:
        """Calculate the total accrued cost based on elapsed time.

        Args:
            hourly_cost: Cost per hour

        Returns:
            Total accrued cost or None if monitoring hasn't started
        """
        # Prefer external timer if available (e.g., trainer's timer that persists across restarts)
        if self._external_timer is not None:
            try:
                # Assume the external timer has a get_elapsed() method
                elapsed_seconds = self._external_timer.get_elapsed()
                if elapsed_seconds is not None:
                    elapsed_hours = elapsed_seconds / 3600.0
                    return hourly_cost * elapsed_hours
            except Exception as e:
                self.logger.debug(f"Failed to get elapsed time from external timer: {e}")

        # Fallback to internal start time
        if self._start_time is None:
            return None

        elapsed_hours = (time.time() - self._start_time) / 3600.0
        return hourly_cost * elapsed_hours
