import logging
import time
from typing import Dict, Optional, Tuple


class Stopwatch:
    """A utility class for timing code execution with support for multiple named timers."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("Stopwatch")
        self._timers: Dict[str, Dict] = {}
        self._global_timer = self._create_timer("__global__")

    def _create_timer(self, name: str) -> Dict:
        """Create a new timer instance."""
        return {
            "name": name,
            "start_time": None,
            "total_elapsed": 0.0,
            "is_running": False,
            "checkpoints": {},  # name -> (time, steps)
        }

    def _get_timer(self, name: Optional[str] = None) -> Dict:
        """Get or create a timer."""
        if name is None:
            return self._global_timer
        if name not in self._timers:
            self._timers[name] = self._create_timer(name)
        return self._timers[name]

    def reset(self, name: Optional[str] = None):
        """Reset timing data for a specific timer or all timers."""
        if name is None:
            self._global_timer = self._create_timer("__global__")
            self._timers.clear()
        else:
            self._timers[name] = self._create_timer(name)

    def start(self, name: Optional[str] = None):
        """Start a timer."""
        timer = self._get_timer(name)
        if timer["is_running"]:
            self.logger.warning(f"Timer '{name or 'global'}' already running")
            return

        timer["start_time"] = time.time()
        timer["is_running"] = True

    def stop(self, name: Optional[str] = None) -> float:
        """Stop a timer and return elapsed time."""
        timer = self._get_timer(name)
        if not timer["is_running"]:
            self.logger.warning(f"Timer '{name or 'global'}' not running")
            return 0.0

        elapsed = time.time() - timer["start_time"]
        timer["total_elapsed"] += elapsed
        timer["is_running"] = False
        return elapsed

    def checkpoint(self, checkpoint_name: str, steps: int, timer_name: Optional[str] = None):
        """Record a named checkpoint with step count."""
        timer = self._get_timer(timer_name)
        if not timer["is_running"]:
            self.logger.warning(f"Timer '{timer_name or 'global'}' not running")
            return

        elapsed = time.time() - timer["start_time"]
        timer["checkpoints"][checkpoint_name] = (elapsed, steps)

    def get_elapsed(self, name: Optional[str] = None) -> float:
        """Get total elapsed time including current run if active."""
        timer = self._get_timer(name)
        if timer["is_running"]:
            return timer["total_elapsed"] + (time.time() - timer["start_time"])
        return timer["total_elapsed"]

    def get_rate(self, steps: int, name: Optional[str] = None, since_start: bool = True) -> float:
        """Calculate rate (steps per second)."""
        timer = self._get_timer(name)

        if since_start:
            elapsed = self.get_elapsed(name)
        else:
            # Get time since last checkpoint
            if not timer["checkpoints"]:
                elapsed = self.get_elapsed(name)
            else:
                last_checkpoint = max(timer["checkpoints"].values(), key=lambda x: x[0])
                elapsed = self.get_elapsed(name) - last_checkpoint[0]

        return steps / elapsed if elapsed > 0 else 0.0

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
        remaining_time, time_str = self.estimate_remaining(current_steps, total_steps, name)

        timer_label = f" [{name}]" if name else ""
        self.logger.info(
            f"{prefix}{timer_label}: {current_steps}/{total_steps} [{rate:.0f} steps/sec] "
            f"({percent:.2f}%) - {time_str} remaining"
        )

    def get_summary(self, name: Optional[str] = None) -> Dict[str, any]:
        """Get summary statistics for a timer."""
        timer = self._get_timer(name)
        return {
            "name": timer["name"],
            "total_elapsed": self.get_elapsed(name),
            "is_running": timer["is_running"],
            "checkpoints": dict(timer["checkpoints"]),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, any]]:
        """Get summaries for all timers."""
        summaries = {}
        if self._global_timer["is_running"] or self._global_timer["total_elapsed"] > 0:
            summaries["global"] = self.get_summary()
        for name in self._timers:
            summaries[name] = self.get_summary(name)
        return summaries
