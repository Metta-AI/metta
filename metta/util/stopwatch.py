import logging
import time
from contextlib import contextmanager
from typing import Any, ContextManager, Dict, Optional, Tuple


class Stopwatch:
    """A utility class for timing code execution with support for multiple named timers."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("Stopwatch")
        self._timers: Dict[str, Dict] = {}
        # Create global timer but don't start it automatically
        self._timers["__global__"] = self._create_timer("__global__")

    def _create_timer(self, name: str) -> Dict:
        """Create a new timer instance."""
        return {
            "name": name,
            "start_time": None,
            "total_elapsed": 0.0,
            "last_elapsed": 0.0,
            "is_running": False,
            "checkpoints": {},  # name -> (time, steps)
        }

    def _get_timer(self, name: Optional[str] = None) -> Dict:
        """Get or create a timer. None defaults to global timer."""
        if name is None:
            name = "__global__"
        if name not in self._timers:
            self._timers[name] = self._create_timer(name)
        return self._timers[name]

    def reset(self, name: Optional[str] = None):
        """Reset timing data for a specific timer or all timers."""
        if name is None:
            # Reset just the global timer
            self._timers["__global__"] = self._create_timer("__global__")
        else:
            self._timers[name] = self._create_timer(name)

    def reset_all(self):
        """Reset all timers including global."""
        self._timers.clear()
        self._timers["__global__"] = self._create_timer("__global__")

    def start(self, name: Optional[str] = None):
        """Start a timer."""
        timer = self._get_timer(name)
        timer_name = name or "global"

        if timer["is_running"]:
            self.logger.warning(f"Timer '{timer_name}' already running")
            return

        timer["start_time"] = time.time()
        timer["is_running"] = True

    def stop(self, name: Optional[str] = None) -> float:
        """Stop a timer and return elapsed time."""
        timer = self._get_timer(name)
        timer_name = name or "global"

        if not timer["is_running"]:
            self.logger.warning(f"Timer '{timer_name}' not running")
            return 0.0

        elapsed = time.time() - timer["start_time"]
        timer["total_elapsed"] += elapsed
        timer["is_running"] = False
        timer["last_elapsed"] = elapsed  # Store last elapsed time
        return elapsed

    @contextmanager
    def time(self, name: Optional[str] = None, log: Optional[int] = None):
        """Context manager for timing a code block.

        Args:
            name: Name of the timer
            log: Optional logging level (e.g., logging.INFO) to automatically log elapsed time on exit

        Usage:
            with stopwatch.time("my_operation", log=logging.INFO):
                # code to time
                pass
        """
        self.start(name)
        try:
            yield self
        finally:
            elapsed = self.stop(name)
            if log is not None:
                display_name = name or "global"
                self.logger.log(log, f"{display_name} took {elapsed:.3f}s")

    def __call__(self, name: Optional[str] = None, log: Optional[int] = None) -> ContextManager["Stopwatch"]:
        """Make Stopwatch callable to return context manager.

        Args:
            name: Name of the timer
            log: Optional logging level (e.g., logging.INFO) to automatically log elapsed time on exit

        Usage:
            with stopwatch("my_operation", log=logging.INFO):
                # code to time
                pass
        """
        return self.time(name, log)

    def checkpoint(self, steps: int, checkpoint_name: Optional[str] = None, timer_name: Optional[str] = None):
        """Record a checkpoint (i.e. lap marker) with step count.

        Args:
            steps: Current step count
            checkpoint_name: Optional name for the checkpoint. If None, uses auto-generated name.
            timer_name: Name of the timer (None for global)

        Usage:
            # Named checkpoint (for specific milestones)
            stopwatch.checkpoint(1000, "epoch_1")

            # Anonymous checkpoint (for lap-based rate tracking)
            stopwatch.checkpoint(1000)
        """
        timer = self._get_timer(timer_name)
        display_name = timer_name or "global"

        if not timer["is_running"]:
            self.logger.warning(f"Timer '{display_name}' not running")
            return

        elapsed = time.time() - timer["start_time"]

        # Generate name if not provided
        if checkpoint_name is None:
            checkpoint_name = f"_lap_{len(timer['checkpoints'])}"

        timer["checkpoints"][checkpoint_name] = (elapsed, steps)

    def lap(self, steps: int, name: Optional[str] = None) -> float:
        """Record a lap and return the lap time.

        Convenience method that creates a checkpoint and returns time since last checkpoint.

        Args:
            steps: Current step count
            name: Timer name (None for global)

        Returns:
            Time elapsed since last lap (or start if first lap)
        """
        timer = self._get_timer(name)

        # Get time since last checkpoint (or start)
        if timer["checkpoints"]:
            last_time, _ = max(timer["checkpoints"].values(), key=lambda x: x[0])
            lap_time = self.get_elapsed(name) - last_time
        else:
            lap_time = self.get_elapsed(name)

        # Record this lap
        self.checkpoint(steps, timer_name=name)

        return lap_time

    def get_elapsed(self, name: Optional[str] = None) -> float:
        """Get total elapsed time including current run if active."""
        timer = self._get_timer(name)
        if timer["is_running"]:
            return timer["total_elapsed"] + (time.time() - timer["start_time"])
        return timer["total_elapsed"]

    def get_last_elapsed(self, name: Optional[str] = None) -> float:
        """Get the elapsed time from the most recent run."""
        timer = self._get_timer(name)
        if timer["is_running"]:
            return time.time() - timer["start_time"]
        return timer["last_elapsed"]

    def get_rate(self, current_steps: int, name: Optional[str] = None) -> float:
        """Calculate average rate (steps per second) since timer start.

        Args:
            current_steps: The current total step count
            name: Timer name (None for global)

        Returns:
            Steps per second since the timer was started
        """
        elapsed = self.get_elapsed(name)
        return current_steps / elapsed if elapsed > 0 else 0.0

    def get_lap_rate(self, current_steps: int, name: Optional[str] = None) -> float:
        """Calculate rate (steps per second) for the current lap.

        A lap is defined as the period since the last checkpoint.
        If no checkpoints exist, calculates rate since timer start.

        Args:
            current_steps: The current total step count
            name: Timer name (None for global)

        Returns:
            Steps per second since the last checkpoint (or since start if no checkpoints)
        """
        timer = self._get_timer(name)

        if not timer["checkpoints"]:
            # No checkpoints, fall back to total rate
            return self.get_rate(current_steps, name)

        # Find the most recent checkpoint
        last_checkpoint_time, last_checkpoint_steps = max(timer["checkpoints"].values(), key=lambda x: x[0])

        # Calculate elapsed time and steps since last checkpoint
        elapsed_since_checkpoint = self.get_elapsed(name) - last_checkpoint_time
        steps_since_checkpoint = current_steps - last_checkpoint_steps

        return steps_since_checkpoint / elapsed_since_checkpoint if elapsed_since_checkpoint > 0 else 0.0

    def format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} sec"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} min"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f} hours"
        else:
            return f"{seconds / 86400:.1f} days"

    def estimate_remaining(self, current_steps: int, total_steps: int, name: Optional[str] = None) -> Tuple[float, str]:
        """Estimate remaining time based on current rate."""
        rate = self.get_rate(current_steps, name)
        if rate <= 0:
            return float("inf"), "unknown"

        remaining_steps = total_steps - current_steps
        remaining_seconds = remaining_steps / rate
        return remaining_seconds, self.format_time(remaining_seconds)

    def log_progress(self, current_steps: int, total_steps: int, name: Optional[str] = None, prefix: str = "Progress"):
        """Log progress with rate and time remaining."""
        rate = self.get_rate(current_steps, name)
        percent = 100.0 * current_steps / total_steps if total_steps > 0 else 0.0
        _remaining_time, time_str = self.estimate_remaining(current_steps, total_steps, name)

        timer_label = f" [{name}]" if name else ""
        self.logger.info(
            f"{prefix}{timer_label}: {current_steps}/{total_steps} [{rate:.0f} steps/sec] "
            f"({percent:.2f}%) - {time_str} remaining"
        )

    def get_summary(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for a timer."""
        timer = self._get_timer(name)
        return {
            "name": timer["name"],
            "total_elapsed": self.get_elapsed(name),
            "is_running": timer["is_running"],
            "checkpoints": dict(timer["checkpoints"]),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all timers."""
        summaries = {}
        for timer_name in self._timers:
            if self._timers[timer_name]["is_running"] or self._timers[timer_name]["total_elapsed"] > 0:
                display_name = "global" if timer_name == "__global__" else timer_name
                summaries[display_name] = self.get_summary(timer_name if timer_name != "__global__" else None)
        return summaries

    def get_all_elapsed(self, exclude_global: bool = True) -> Dict[str, float]:
        """Get elapsed times for all timers.

        Args:
            exclude_global: If True, excludes the global timer from results

        Returns:
            Dictionary mapping timer names to elapsed times in seconds
        """
        results = {}
        for timer_name in self._timers:
            if exclude_global and timer_name == "__global__":
                continue

            elapsed = self.get_elapsed(timer_name if timer_name != "__global__" else None)

            results[timer_name] = elapsed

        return results
