import functools
import inspect
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, Final, List, Optional, Tuple, TypedDict, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class TimerReference(TypedDict):
    """Reference to where a timer was used."""

    filename: str
    lineno: int


class Checkpoint(TypedDict):
    """A checkpoint/lap marker in a timer."""

    elapsed_time: float
    steps: int


@dataclass
class Timer:
    """State and statistics for a single timer."""

    name: str
    start_time: Optional[float] = None
    total_elapsed: float = 0.0
    last_elapsed: float = 0.0
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)
    lap_counter: int = 0
    references: List[TimerReference] = field(default_factory=list)

    def is_running(self) -> bool:
        return self.start_time is not None


def with_timer(timer: "Stopwatch", timer_name: str, log_level: Optional[int] = None):
    """Decorator that wraps function execution in a timer context.

    Args:
        timer: The Stopwatch instance to use
        timer_name: Name of the timer
        log_level: Optional logging level to automatically log elapsed time

    Usage:
        @with_timer(my_timer, "reset")
        def reset(self, seed=None):
            # method content
            pass
    """

    def decorator(func: F) -> F:
        # Capture where the decorator is applied
        frame = inspect.currentframe()
        if frame and frame.f_back:
            filename = frame.f_back.f_code.co_filename
            lineno = frame.f_back.f_lineno
        else:
            filename = "unknown"
            lineno = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer.time(timer_name, log_level=log_level, filename=filename, lineno=lineno):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def with_instance_timer(timer_name: str, log_level: Optional[int] = None, timer_attr: str = "timer"):
    """Decorator that uses a timer from the instance.

    Args:
        timer_name: Name of the timer
        log_level: Optional logging level
        timer_attr: Name of the timer attribute on the instance (default: "timer")

    Usage:
        class MyClass:
            def __init__(self):
                self.timer = Stopwatch()

            @with_instance_timer("method_timer")
            def my_method(self, value):
                # method content
                return value
    """

    def decorator(func: F) -> F:
        # Capture where the decorator is applied
        frame = inspect.currentframe()
        if frame and frame.f_back:
            filename = frame.f_back.f_code.co_filename
            lineno = frame.f_back.f_lineno
        else:
            filename = "unknown"
            lineno = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First argument should be 'self' for instance methods
            if not args:
                raise ValueError("with_instance_timer can only be used on instance methods")
            instance = args[0]
            timer = getattr(instance, timer_attr)
            with timer.time(timer_name, log_level=log_level, filename=filename, lineno=lineno):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


class Stopwatch:
    """A utility class for timing code execution with support for multiple named timers."""

    _GLOBAL_TIMER_NAME: Final[str] = "global"  # Reserved name for the global timer

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("Stopwatch")
        self._timers: Dict[str, Timer] = {}
        # Create global timer but don't start it automatically
        self._timers[self.GLOBAL_TIMER_NAME] = self._create_timer(self.GLOBAL_TIMER_NAME)

    @property
    def GLOBAL_TIMER_NAME(self) -> str:
        """Read-only access to the global timer name."""
        return self._GLOBAL_TIMER_NAME

    def _create_timer(self, name: str) -> Timer:
        """Create a new timer instance."""
        return Timer(
            name=name,
            start_time=None,
            total_elapsed=0.0,
            last_elapsed=0.0,
            checkpoints={},
            lap_counter=0,
            references=[],
        )

    def _get_timer(self, name: Optional[str] = None) -> Timer:
        """Get or create a timer. None defaults to global timer."""
        if name == self.GLOBAL_TIMER_NAME:
            raise ValueError(
                f"'{self.GLOBAL_TIMER_NAME}' is a reserved timer name. Use None to access the global timer."
            )

        if name is None:
            name = self.GLOBAL_TIMER_NAME

        if name not in self._timers:
            self._timers[name] = self._create_timer(name)

        return self._timers[name]

    def _capture_caller_info(self, skip_frames: int = 2) -> Tuple[str, int]:
        """Capture the filename and line number of the caller.

        Args:
            skip_frames: Number of stack frames to skip (default 2: this method + caller)

        Returns:
            Tuple of (filename, line_number)
        """
        frame = inspect.currentframe()
        try:
            # Skip the specified number of frames
            for _ in range(skip_frames):
                if frame is None:
                    break
                frame = frame.f_back

            if frame is not None:
                return frame.f_code.co_filename, frame.f_lineno
            return "unknown", 0
        finally:
            # Avoid reference cycles
            del frame

    def reset(self, name: Optional[str] = None):
        """Reset timing data for a specific timer or all timers."""
        if name is None:
            # Reset just the global timer
            self._timers[self.GLOBAL_TIMER_NAME] = self._create_timer(self.GLOBAL_TIMER_NAME)
        else:
            if name == self.GLOBAL_TIMER_NAME:
                raise ValueError(f"Use None to reset the global timer, not '{self.GLOBAL_TIMER_NAME}'")
            self._timers[name] = self._create_timer(name)

    def reset_all(self):
        """Reset all timers including global."""
        self._timers.clear()
        self._timers[self.GLOBAL_TIMER_NAME] = self._create_timer(self.GLOBAL_TIMER_NAME)

    def start(self, name: Optional[str] = None, filename: Optional[str] = None, lineno: Optional[int] = None):
        """Start a timer.

        Args:
            name: Timer name
            filename: Optional filename for reference (auto-captured if not provided)
            lineno: Optional line number for reference (auto-captured if not provided)
        """
        timer = self._get_timer(name)
        timer_name = name or "global"

        if timer.is_running():
            self.logger.warning(f"Timer '{timer_name}' already running")
            return

        # Capture caller info if not provided
        if filename is None or lineno is None:
            filename, lineno = self._capture_caller_info(skip_frames=2)

        # Store reference
        timer.references.append(TimerReference(filename=filename, lineno=lineno))

        timer.start_time = time.time()

    def stop(self, name: Optional[str] = None) -> float:
        """Stop a timer and return elapsed time."""
        timer = self._get_timer(name)
        timer_name = name or "global"

        if not timer.is_running():
            self.logger.warning(f"Timer '{timer_name}' not running")
            return 0.0

        if timer.start_time is None:
            self.logger.warning(f"Timer '{timer_name}' has no start time")
            return 0.0

        elapsed = time.time() - timer.start_time
        timer.total_elapsed += elapsed
        timer.last_elapsed = elapsed
        timer.start_time = None
        return elapsed

    @contextmanager
    def time(
        self,
        name: Optional[str] = None,
        log_level: Optional[int] = None,
        filename: Optional[str] = None,
        lineno: Optional[int] = None,
    ):
        """Context manager for timing a code block.

        Args:
            name: Name of the timer
            log_level: Optional logging level (e.g., logging.INFO) to automatically log elapsed time on exit
            filename: Optional filename for reference (auto-captured if not provided)
            lineno: Optional line number for reference (auto-captured if not provided)

        Usage:
            with stopwatch.time("my_operation", log_level=logging.INFO):
                # code to time
                pass
        """
        # Capture caller info if not provided
        if filename is None or lineno is None:
            caller_filename, caller_lineno = self._capture_caller_info(skip_frames=3)
            filename = filename or caller_filename
            lineno = lineno or caller_lineno

        self.start(name, filename=filename, lineno=lineno)
        try:
            yield self
        finally:
            elapsed = self.stop(name)
            if log_level is not None:
                display_name = name or "global"
                self.logger.log(log_level, f"{display_name} took {elapsed:.3f}s")

    def __call__(self, name: Optional[str] = None, log_level: Optional[int] = None) -> ContextManager["Stopwatch"]:
        """Make Stopwatch callable to return context manager.

        Args:
            name: Name of the timer
            log_level: Optional logging level (e.g., logging.INFO) to automatically log elapsed time on exit

        Usage:
            with stopwatch("my_operation", log_level=logging.INFO):
                # code to time
                pass
        """
        return self.time(name, log_level)

    def checkpoint(
        self, steps: Optional[int] = None, checkpoint_name: Optional[str] = None, timer_name: Optional[str] = None
    ):
        """Record a checkpoint (i.e. lap marker) with step count.

        Args:
            steps: Step count. If None, uses internal lap counter
            checkpoint_name: Optional name for the checkpoint. If None, uses auto-generated name.
            timer_name: Name of the timer (None for global)

        Usage:
            # With explicit steps
            stopwatch.checkpoint(1000, "epoch_1")

            # With auto-incrementing internal counter
            stopwatch.checkpoint()  # uses internal counter
        """
        timer = self._get_timer(timer_name)

        if timer.start_time is not None:
            elapsed = time.time() - timer.start_time
        else:
            # For stopped timers, use total elapsed time
            elapsed = timer.total_elapsed

        # Use internal counter if steps not provided
        if steps is None:
            timer.lap_counter += 1
            steps = timer.lap_counter

        # Generate name if not provided
        if checkpoint_name is None:
            checkpoint_name = f"_lap_{len(timer.checkpoints)}"

        timer.checkpoints[checkpoint_name] = Checkpoint(elapsed_time=elapsed, steps=steps)

    def lap(self, steps: Optional[int] = None, name: Optional[str] = None) -> float:
        """Record a lap and return the lap time.

        Convenience method that creates a checkpoint and returns time since last checkpoint.

        Args:
            steps: Step count. If None, uses internal lap counter
            name: Timer name (None for global)

        Returns:
            Time elapsed since last lap (or start if first lap)
        """
        timer = self._get_timer(name)

        # Get time since last checkpoint (or start)
        if timer.checkpoints:
            last_time = max(timer.checkpoints.values(), key=lambda x: x["elapsed_time"])["elapsed_time"]
            lap_time = self.get_elapsed(name) - last_time
        else:
            lap_time = self.get_elapsed(name)

        # Record this lap
        self.checkpoint(steps, timer_name=name)

        return lap_time

    def checkpoint_all(self, steps: Optional[int] = None, checkpoint_name: Optional[str] = None):
        """Record a checkpoint on all active timers.

        Args:
            steps: Step count. If None, uses internal lap counter for each timer
            checkpoint_name: Optional name for the checkpoint
        """
        for timer_name, _timer in self._timers.items():
            actual_name = timer_name if timer_name != self.GLOBAL_TIMER_NAME else None
            self.checkpoint(steps, checkpoint_name, actual_name)

    def lap_all(self, steps: Optional[int] = None, exclude_global: bool = False) -> Dict[str, float]:
        """Mark a lap on all timers and return lap times.

        Args:
            steps: Step count. If None, uses internal lap counter for each timer
            exclude_global: If True, excludes the global timer from results (but still records lap)

        Returns:
            Dictionary mapping timer names to their lap times
        """
        lap_times = {}
        for timer_name, _timer in self._timers.items():
            # Use None for global timer in internal API
            actual_name = None if timer_name == self.GLOBAL_TIMER_NAME else timer_name
            lap_time = self.lap(steps, actual_name)

            # Only include in results if not excluding global or not global timer
            if not (exclude_global and timer_name == self.GLOBAL_TIMER_NAME):
                lap_times[timer_name] = lap_time
        return lap_times

    def get_elapsed(self, name: Optional[str] = None) -> float:
        """Get total elapsed time including current run if active."""
        timer = self._get_timer(name)
        if timer.start_time is not None:
            return timer.total_elapsed + (time.time() - timer.start_time)
        return timer.total_elapsed

    def get_last_elapsed(self, name: Optional[str] = None) -> float:
        """Get the elapsed time from the most recent run."""
        timer = self._get_timer(name)
        if timer.start_time is not None:
            return time.time() - timer.start_time
        return timer.last_elapsed

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

        if not timer.checkpoints:
            # No checkpoints, fall back to total rate
            return self.get_rate(current_steps, name)

        # Find the most recent checkpoint
        last_checkpoint = max(timer.checkpoints.values(), key=lambda x: x["elapsed_time"])

        # Calculate elapsed time and steps since last checkpoint
        elapsed_since_checkpoint = self.get_elapsed(name) - last_checkpoint["elapsed_time"]
        steps_since_checkpoint = current_steps - last_checkpoint["steps"]

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
            "name": timer.name,
            "total_elapsed": self.get_elapsed(name),
            "is_running": timer.is_running(),
            "checkpoints": dict(timer.checkpoints),
            "references": timer.references.copy(),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all timers."""
        summaries = {}
        for timer_name, timer in self._timers.items():
            if timer.is_running() or timer.total_elapsed > 0:
                actual_name = None if timer_name == self.GLOBAL_TIMER_NAME else timer_name
                summaries[timer_name] = self.get_summary(actual_name)
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
            if exclude_global and timer_name == self.GLOBAL_TIMER_NAME:
                continue

            actual_name = None if timer_name == self.GLOBAL_TIMER_NAME else timer_name
            elapsed = self.get_elapsed(actual_name)
            results[timer_name] = elapsed

        return results

    def get_lap_steps(self, lap_index: int = -1, name: Optional[str] = None) -> Optional[int]:
        """Get the step count for a specific lap.

        Args:
            lap_index: Index of the lap (1-based). Negative indices count from the end.
                    default -1 is the most recent completed lap
            name: Timer name (None for global)

        Returns:
            Step count for the specified lap, or None if the lap doesn't exist.
            For lap N, this returns the number of steps taken between checkpoint N-1 and checkpoint N.
        """
        timer = self._get_timer(name)

        if not timer.checkpoints:
            return None

        # Sort checkpoints by time to get them in order
        sorted_checkpoints = sorted(timer.checkpoints.items(), key=lambda x: x[1]["elapsed_time"])

        # Need at least 2 checkpoints to have a completed lap
        if len(sorted_checkpoints) < 2:
            return None

        # Handle negative indices
        if lap_index < 0:
            lap_index = len(sorted_checkpoints) + lap_index + 1

        if lap_index < 1 or lap_index > len(sorted_checkpoints):
            return None

        # Get the step count at the end of the requested lap
        _, end_checkpoint = sorted_checkpoints[lap_index - 1]

        # Get the step count at the start of the lap (or 0 if it's the first lap)
        if lap_index > 1:
            _, start_checkpoint = sorted_checkpoints[lap_index - 2]
            start_steps = start_checkpoint["steps"]
        else:
            start_steps = 0

        return end_checkpoint["steps"] - start_steps

    def get_filename(self, name: Optional[str] = None) -> str:
        """Get a file reference for where this timer is used

        Args:
            name: Timer name (None for global)

        Returns:
            Filename where timer was used, or "multifile" if used in multiple files
        """
        timer = self._get_timer(name)
        if not timer.references:
            return "unknown"

        # Get unique filenames efficiently
        first_file = timer.references[0]["filename"]

        # Check if all references are from the same file
        for ref in timer.references[1:]:
            if ref["filename"] != first_file:
                return "multifile"

        return first_file
