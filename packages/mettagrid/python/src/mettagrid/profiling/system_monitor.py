import logging
import os
import time
from collections import deque
from threading import Lock, Thread
from typing import Any, Callable

import psutil
import torch
from typing_extensions import TypeVar

T = TypeVar("T")


class SystemMonitor:
    def __init__(
        self,
        sampling_interval_sec: float = 1.0,
        history_size: int = 100,
        log_level: int | None = None,
        auto_start: bool = True,
        external_timer: Any | None = None,
    ):
        self.logger = logging.getLogger(f"SystemMonitor.{id(self)}")
        if log_level is None:
            self.logger.disabled = True
        else:
            self.logger.setLevel(log_level)

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

    def stats(self) -> dict[str, float]:
        stats = {}
        summary = self.get_summary()
        for metric_name, metric_data in summary["metrics"].items():
            if metric_data["latest"] is not None:
                stats[f"monitor/{metric_name}"] = metric_data["latest"]
        return stats

    def get_summary(self) -> dict[str, Any]:
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

    def get_history(self, metric: str) -> list[tuple[float, Any]]:
        with self._lock:
            if metric not in self._metrics:
                return []
            return list(self._metrics[metric])

    def get_latest(self, metric: str | None = None) -> Any:
        with self._lock:
            if metric:
                return self._latest.get(metric)
            return self._latest.copy()

    def _initialize_default_metrics(self):
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
            "process_memory_mb": lambda: psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
            "process_cpu_percent": lambda: self._process.cpu_percent(),
            "process_threads": lambda: psutil.Process().num_threads(),
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
                        f"gpu{i}_utilization": lambda idx=i: torch.cuda.utilization(idx),
                        f"gpu{i}_memory_percent": lambda idx=i: self._get_single_gpu_memory_percent(idx),
                        f"gpu{i}_memory_used_mb": lambda idx=i: self._get_single_gpu_memory_used_mb(idx),
                    }
                )

            self.logger.info(f"GPU monitoring enabled via CUDA ({gpu_count} devices)")

        # Initialize history storage for all metrics
        for name in self._metric_collectors:
            self._metrics[name] = deque(maxlen=self.history_size)

    def _get_cpu_temperature(self) -> float | None:
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

    def _get_gpu_utilization_cuda(self) -> float | None:
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
                except ZeroDivisionError:
                    # In case total memory is reported as 0 (shouldn't happen but defensive programming)
                    self.logger.warning(f"GPU {i} reports 0 total memory")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error getting GPU {i} memory: {type(e).__name__}: {e}")
                    continue
            return sum(percents) / len(percents) if percents else None
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
                except Exception as e:
                    self.logger.warning(f"Unexpected error getting GPU {i} memory: {type(e).__name__}: {e}")
                    continue
            return total_used if count > 0 else None
        except Exception as e:
            self.logger.warning(f"Unexpected error in GPU memory used: {type(e).__name__}: {e}")
            return None

    def _get_single_gpu_memory_percent(self, gpu_idx: int) -> float | None:
        free, total = torch.cuda.mem_get_info(gpu_idx)
        if total > 0:
            return (total - free) / total * 100
        return None

    def _get_single_gpu_memory_used_mb(self, gpu_idx: int) -> float | None:
        free, total = torch.cuda.mem_get_info(gpu_idx)
        return (total - free) / (1024 * 1024)

    def _collect_sample(self) -> None:
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
        self.logger.debug("Monitor thread started")

        while not self._stop_flag:
            self._collect_sample()
            time.sleep(self.sampling_interval_sec)

        self.logger.debug("Monitor thread stopped")

    def start(self) -> None:
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
        if not self._thread or not self._thread.is_alive():
            return

        self._stop_flag = True
        self._thread.join(timeout=self.sampling_interval_sec * 2)
        self.logger.info("System monitoring stopped")

    def _calculate_accrued_cost(self, hourly_cost: float) -> float | None:
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
