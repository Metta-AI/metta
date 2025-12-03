from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, ContextManager, Final, Tuple, TypedDict, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class Checkpoint(TypedDict):
    """A checkpoint/lap marker in a timer."""

    elapsed_time: float
    steps: int


@dataclass
class Timer:
    """State and statistics for a single timer."""

    name: str
    start_time: float | None = None
    total_elapsed: float = 0.0
    last_elapsed: float = 0.0
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    lap_counter: int = 0
    references: set[tuple[str, int]] = field(default_factory=set)  # Changed to set of tuples
    max_laps: int = 4

    def is_running(self) -> bool:
        return self.start_time is not None

    def cleanup_old_checkpoints(self):
        """Remove oldest checkpoints if we exceed max_laps limit."""
        max_checkpoints = self.max_laps + 1
        if len(self.checkpoints) > max_checkpoints:
            # Since checkpoints are added in chronological order and dicts maintain
            # insertion order (Python 3.7+), we can just keep the last max_checkpoints items
            checkpoint_items = list(self.checkpoints.items())
            checkpoints_to_keep = checkpoint_items[-max_checkpoints:]
            self.checkpoints = dict(checkpoints_to_keep)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timer":
        """Create a Timer instance from a dictionary created using .asdict()."""
        references_data = data.get("references", [])

        # Convert list of tuples back to set (from serialization)
        references = set(references_data) if references_data else set()

        return cls(
            name=data["name"],
            start_time=data.get("start_time"),
            total_elapsed=data.get("total_elapsed", 0.0),
            last_elapsed=data.get("last_elapsed", 0.0),
            checkpoints=data.get("checkpoints", {}),
            lap_counter=data.get("lap_counter", 0),
            references=references,
            max_laps=data.get("max_laps", 4),
        )


def _capture_caller_info(extra_skip_frames: int = 0) -> Tuple[str, int]:
    """Capture the filename and line number of the caller."""
    frame = inspect.currentframe()
    try:
        # Skip baseline frames (this function + direct caller) plus any extra
        for _ in range(2 + extra_skip_frames):
            if frame is None:
                break
            frame = frame.f_back

        if frame is not None:
            return frame.f_code.co_filename, frame.f_lineno
        return "unknown", 0
    finally:
        # Avoid reference cycles
        del frame


def with_timer(timer: Stopwatch, name: str, log_level: int | None = None):
    """Decorator that wraps function execution in a timer context."""

    def decorator(func: F) -> F:
        # Capture where the decorator is applied
        filename, lineno = _capture_caller_info(extra_skip_frames=0)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer.time(name, log_level=log_level, filename=filename, lineno=lineno):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def with_instance_timer(name: str, log_level: int | None = None, timer_attr: str = "timer"):
    """Decorator that uses a timer from the instance."""

    def decorator(func: F) -> F:
        # Capture where the decorator is applied
        filename, lineno = _capture_caller_info(extra_skip_frames=0)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First argument should be 'self' for instance methods
            if not args:
                raise ValueError("with_instance_timer can only be used on instance methods")
            instance = args[0]
            timer = getattr(instance, timer_attr)
            with timer.time(name, log_level=log_level, filename=filename, lineno=lineno):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def with_lock(func: F) -> F:
    """Decorator that acquires the instance lock before executing the method.

    Usage:
        @with_lock
        def my_method(self, ...):
            # method content - automatically thread-safe
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return cast(F, wrapper)


class Stopwatch:
    """A thread-safe utility class for timing code execution with support for multiple named timers."""

    _GLOBAL_TIMER_NAME: Final[str] = "global"  # Reserved name for the global timer

    def __init__(self, log_level: int | None = None, max_laps: int = 4):
        self.logger = logging.getLogger(f"Stopwatch.{id(self)}")
        if log_level is None:
            self.logger.disabled = True
        else:
            self.logger.setLevel(log_level)

        self.max_laps = max_laps
        self._timers: dict[str, Timer] = {}
        # Create global timer but don't start it automatically
        self._timers[self.GLOBAL_TIMER_NAME] = self._create_timer(self.GLOBAL_TIMER_NAME)
        # Add a lock for thread safety
        self._lock = threading.RLock()  # RLock allows recursive locking

    @property
    def GLOBAL_TIMER_NAME(self) -> str:
        """Read-only access to the global timer name."""
        return self._GLOBAL_TIMER_NAME

    def _create_timer(self, name: str) -> Timer:
        """Create a new timer instance."""
        timer = Timer(
            name=name,
            start_time=None,
            total_elapsed=0.0,
            last_elapsed=0.0,
            checkpoints={},
            lap_counter=0,
            references=set(),  # Initialize as empty set
            max_laps=self.max_laps,
        )

        # Add initial checkpoint at step 0 to serve as a reference point for lap calculations
        timer.checkpoints["_start"] = Checkpoint(elapsed_time=0.0, steps=0)

        return timer

    def _get_timer(self, name: str | None = None) -> Timer:
        """Get or create a timer. None defaults to global timer."""

        name = name or self.GLOBAL_TIMER_NAME

        with self._lock:
            if name not in self._timers:
                self._timers[name] = self._create_timer(name)
            return self._timers[name]

    @with_lock
    def reset(self, name: str | None = None):
        """Reset timing data while preserving timer existence and reference history."""

        name = name or self.GLOBAL_TIMER_NAME

        if name in self._timers:
            timer = self._timers[name]
            # Reset state while preserving the timer's existence and references
            timer.start_time = None
            timer.total_elapsed = 0.0
            timer.last_elapsed = 0.0
            timer.checkpoints.clear()
            timer.lap_counter = 0
            timer.max_laps = self.max_laps
            # Keep timer.references intact to preserve decorator information
            timer.checkpoints["_reset"] = Checkpoint(elapsed_time=0.0, steps=0)
        else:
            # Timer doesn't exist yet, create it (with a "_start_" checkpoint)
            self._timers[name] = self._create_timer(name)

    def reset_all(self):
        """Reset all timers while preserving timer existence and reference history."""
        with self._lock:
            timer_names = list(self._timers.keys())

        for name in timer_names:
            self.reset(name)

    @with_lock
    def start(
        self,
        name: str | None = None,
        filename: str | None = None,
        lineno: int | None = None,
    ):
        """Start a timer with optional reference tracking."""
        timer = self._get_timer(name)
        name = name or self.GLOBAL_TIMER_NAME

        if timer.is_running():
            self.logger.warning(f"Timer '{name}' already running")
            return

        # Capture caller info if not provided
        if filename is None or lineno is None:
            caller_filename, caller_lineno = _capture_caller_info(extra_skip_frames=1)  # skip lock
            filename = filename or caller_filename
            lineno = lineno or caller_lineno

        # Store reference as tuple - set automatically handles deduplication
        timer.references.add((filename, lineno))
        timer.start_time = time.time()

    @with_lock
    def stop(self, name: str | None = None) -> float:
        """Stop a timer and return elapsed time"""
        timer = self._get_timer(name)
        name = name or self.GLOBAL_TIMER_NAME

        if not timer.is_running():
            self.logger.warning(f"Timer '{name}' not running")
            return 0.0

        if timer.start_time is None:
            self.logger.warning(f"Timer '{name}' has no start time")
            return 0.0

        elapsed = time.time() - timer.start_time
        timer.total_elapsed += elapsed
        timer.last_elapsed = elapsed
        timer.start_time = None
        return elapsed

    @contextmanager
    def time(
        self,
        name: str | None = None,
        log_level: int | None = None,
        filename: str | None = None,
        lineno: int | None = None,
    ):
        """Context manager for timing a code block with optional automatic logging."""

        # Capture caller info if not provided
        if filename is None or lineno is None:
            caller_filename, caller_lineno = _capture_caller_info(extra_skip_frames=1)  # skip context
            filename = filename or caller_filename
            lineno = lineno or caller_lineno

        self.start(name, filename=filename, lineno=lineno)
        try:
            yield self
        finally:
            elapsed = self.stop(name)
            if log_level is not None:
                display_name = name or self.GLOBAL_TIMER_NAME
                self.logger.log(log_level, f"{display_name} took {elapsed:.3f}s")

    def __call__(self, name: str | None = None, log_level: int | None = None) -> ContextManager[Stopwatch]:
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

    @with_lock
    def checkpoint(
        self,
        steps: int | None = None,
        checkpoint_name: str | None = None,
        name: str | None = None,
    ):
        """Record a checkpoint (i.e. lap marker) with step count.

        Args:
            steps: Step count. If None, uses internal lap counter
            checkpoint_name: Optional name for the checkpoint. If None, uses auto-generated name.
            name: Name of the timer (None for global)

        Usage:
            # With explicit steps
            stopwatch.checkpoint(1000, "epoch_1")

            # With auto-incrementing internal counter
            stopwatch.checkpoint()  # uses internal counter
        """
        timer = self._get_timer(name)

        elapsed = self.get_elapsed(name)
        timer.lap_counter += 1

        # Use internal counter if steps not provided
        if steps is None:
            steps = timer.lap_counter

        # Generate name if not provided
        if checkpoint_name is None:
            # Use 1-based indexing to match lap numbers
            checkpoint_name = f"_lap_{timer.lap_counter}"

        timer.checkpoints[checkpoint_name] = Checkpoint(elapsed_time=elapsed, steps=steps)
        timer.cleanup_old_checkpoints()

    def checkpoint_all(self, steps: int | None = None, checkpoint_name: str | None = None):
        """Record a checkpoint on all active timers.

        Args:
            steps: Step count. If None, uses internal lap counter for each timer
            checkpoint_name: Optional name for the checkpoint
        """
        with self._lock:
            timer_names = list(self._timers.keys())

        for name in timer_names:
            self.checkpoint(steps, checkpoint_name, name)

    @with_lock
    def lap(self, steps: int | None = None, name: str | None = None) -> float:
        """Record a lap and return the lap time.

        Convenience method that creates a checkpoint and returns time since last checkpoint.

        Args:
            steps: Step count. If None, uses internal lap counter
            name: Timer name (None for global)

        Returns:
            Time elapsed since last lap (or start if first lap)
        """

        timer = self._get_timer(name)
        current_elapsed = self.get_elapsed(name)

        # Get time since last checkpoint (or start)
        if timer.checkpoints:
            # Since checkpoints are added in chronological order and dicts maintain
            # insertion order (Python 3.7+), we can get the last one efficiently
            *_, last_item = timer.checkpoints.items()
            _last_checkpoint_name, last_checkpoint = last_item
            lap_time = current_elapsed - last_checkpoint["elapsed_time"]
        else:
            lap_time = current_elapsed

        # Record this lap (still within the lock)
        self.checkpoint(steps, name=name)

        return lap_time

    def lap_all(self, steps: int | None = None, exclude_global: bool = True) -> dict[str, float]:
        """Mark a lap on all timers and return lap times.

        Args:
            steps: Step count. If None, uses internal lap counter for each timer
            exclude_global: If True, excludes the global timer from results (but still records lap)

        Returns:
            Dictionary mapping timer names to their lap times
        """
        with self._lock:
            timer_names = list(self._timers.keys())

        lap_times = {name: self.lap(steps, name) for name in timer_names}

        if exclude_global:
            lap_times.pop(self.GLOBAL_TIMER_NAME)

        return lap_times

    @with_lock
    def get_elapsed(self, name: str | None = None) -> float:
        """Get total elapsed time including current run if active."""
        timer = self._get_timer(name)
        if timer.start_time is not None:
            return timer.total_elapsed + (time.time() - timer.start_time)
        return timer.total_elapsed

    @with_lock
    def get_last_elapsed(self, name: str | None = None) -> float:
        """Get the elapsed time from the most recent run."""
        timer = self._get_timer(name)
        if timer.start_time is not None:
            return time.time() - timer.start_time
        return timer.last_elapsed

    @with_lock
    def get_rate(self, current_steps: int, name: str | None = None) -> float:
        """Calculate average rate (steps per second) since timer start.

        Args:
            current_steps: The current total step count
            name: Timer name (None for global)

        Returns:
            Steps per second since the timer was started
        """
        elapsed = self.get_elapsed(name)
        return current_steps / elapsed if elapsed > 0 else 0.0

    @with_lock
    def get_lap_rate(self, current_steps: int, name: str | None = None) -> float:
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

        # Get the most recent checkpoint efficiently (same as in lap())
        *_, last_item = timer.checkpoints.items()
        _last_checkpoint_name, last_checkpoint = last_item

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

    def estimate_remaining(self, current_steps: int, total_steps: int, name: str | None = None) -> Tuple[float, str]:
        """Estimate remaining time based on current rate."""
        rate = self.get_rate(current_steps, name)
        if rate <= 0:
            return float("inf"), "unknown"

        remaining_steps = total_steps - current_steps
        remaining_seconds = remaining_steps / rate
        return remaining_seconds, self.format_time(remaining_seconds)

    def log_progress(
        self,
        current_steps: int,
        total_steps: int,
        name: str | None = None,
        prefix: str = "Progress",
    ):
        """Log progress with rate and time remaining."""
        rate = self.get_rate(current_steps, name)
        percent = 100.0 * current_steps / total_steps if total_steps > 0 else 0.0
        _remaining_time, time_str = self.estimate_remaining(current_steps, total_steps, name)

        timer_label = f" [{name}]" if name else ""
        self.logger.info(
            f"{prefix}{timer_label}: {current_steps}/{total_steps} [{rate:.0f} steps/sec] "
            f"({percent:.2f}%) - {time_str} remaining"
        )

    @with_lock
    def get_summary(self, name: str | None = None) -> dict[str, Any]:
        """Get summary statistics for a timer."""
        timer = self._get_timer(name)
        return {
            "name": timer.name,
            "total_elapsed": self.get_elapsed(name),
            "is_running": timer.is_running(),
            "checkpoints": dict(timer.checkpoints),
            "references": list(timer.references),  # Convert set to list for output
            "max_laps": timer.max_laps,
        }

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get summaries for all timers."""
        with self._lock:
            timer_items = list(self._timers.items())

        summaries = {}
        for name, timer in timer_items:
            if timer.is_running() or timer.total_elapsed > 0:
                summaries[name] = self.get_summary(name)
        return summaries

    def get_all_elapsed(self, exclude_global: bool = True) -> dict[str, float]:
        """Get elapsed times for all timers.

        Args:
            exclude_global: If True, excludes the global timer from results

        Returns:
            Dictionary mapping timer names to elapsed times in seconds
        """
        with self._lock:
            timer_names = list(self._timers.keys())

        results = {name: self.get_elapsed(name) for name in timer_names}

        if exclude_global:
            results.pop(self.GLOBAL_TIMER_NAME, None)

        return results

    def _get_lap_checkpoints(
        self, lap_index: int = -1, name: str | None = None
    ) -> Tuple[Checkpoint, Checkpoint] | None:
        """Get the start and end checkpoints for a specified lap.

        Args:
            lap_index: Index of the lap (negative only, counting from the end).
                       Must be between -1 and -max_laps (inclusive)
                       default -1 is the most recent completed lap
            name: Timer name (None for global)

        Returns:
            The start and end checkpoints for the lap, or None if the lap doesn't exist.
            For lap N, this returns checkpoint N-1 and checkpoint N.

        """
        timer = self._get_timer(name)

        if not timer.checkpoints:
            return None

        # Checkpoints are already in chronological order due to insertion order
        # Need at least 2 checkpoints to have a lap
        checkpoint_items = list(timer.checkpoints.items())
        if len(checkpoint_items) < 2:
            return None

        # Only accept negative indices within max_laps range
        if lap_index >= 0 or lap_index < -timer.max_laps:
            return None

        # Calculate start and end checkpoint indices for the given lap_index
        # lap_index = -1 means the most recent lap (between last two checkpoints)
        # lap_index = -2 means second most recent lap, etc.

        end_checkpoint_idx = len(checkpoint_items) + lap_index  # Convert negative to array index
        start_checkpoint_idx = end_checkpoint_idx - 1

        # Validate indices are within bounds
        if end_checkpoint_idx < 0 or end_checkpoint_idx >= len(checkpoint_items):
            return None
        if start_checkpoint_idx < 0:
            return None

        # Get the checkpoints
        _, end_checkpoint = checkpoint_items[end_checkpoint_idx]
        _, start_checkpoint = checkpoint_items[start_checkpoint_idx]

        return start_checkpoint, end_checkpoint

    @with_lock
    def get_lap_time(self, lap_index: int = -1, name: str | None = None) -> float | None:
        checkpoints = self._get_lap_checkpoints(lap_index, name)
        if checkpoints is None:
            return None
        start_checkpoint, end_checkpoint = checkpoints
        return end_checkpoint["elapsed_time"] - start_checkpoint["elapsed_time"]

    @with_lock
    def get_lap_steps(self, lap_index: int = -1, name: str | None = None) -> int | None:
        checkpoints = self._get_lap_checkpoints(lap_index, name)
        if checkpoints is None:
            return None
        start_checkpoint, end_checkpoint = checkpoints
        return end_checkpoint["steps"] - start_checkpoint["steps"]

    @with_lock
    def get_filename(self, name: str | None = None) -> str:
        """Get a file reference for where this timer is used

        Args:
            name: Timer name (None for global)

        Returns:
            Filename where timer was used, or "multifile" if used in multiple files
        """
        timer = self._get_timer(name)
        if not timer.references:
            return "unknown"

        # Get unique filenames from the set of references
        filenames = {filename for filename, _ in timer.references}

        if len(filenames) == 1:
            return next(iter(filenames))
        else:
            return "multifile"

    @with_lock
    def save_state(self) -> dict[str, Any]:
        """Save the complete state of all timers to a serializable dictionary.

        Returns:
            Dictionary containing all timer states that can be serialized with pickle/json
        """
        state = {
            "version": "1.0",  # Version for future compatibility
            "timers": {},
            "max_laps": self.max_laps,
        }

        current_time = time.time()

        for name, timer in self._timers.items():
            # Convert timer to dict using dataclass asdict
            timer_dict = asdict(timer)

            # Convert set to list for JSON serialization
            timer_dict["references"] = list(timer.references)

            if timer.is_running() and timer.start_time is not None:
                # Calculate elapsed time up to this point
                elapsed_since_start = current_time - timer.start_time
                timer_dict["total_elapsed"] += elapsed_since_start
                # Store how long it was running when saved
                timer_dict["_was_running_for"] = elapsed_since_start
                # Mark that it was running
                timer_dict["_was_running"] = True
                # Clear start_time in the saved state since we've added the elapsed time
                timer_dict["start_time"] = None
            else:
                timer_dict["_was_running"] = False

            state["timers"][name] = timer_dict

        return state

    @with_lock
    def load_state(self, state: dict[str, Any], resume_running: bool = True) -> None:
        """Load timer state from a dictionary.

        Args:
            state: Dictionary containing timer state (from save_state())
            resume_running: If True, timers that were running when saved will be resumed
        """

        current_time = time.time()

        if not isinstance(state, dict) or "timers" not in state:
            raise ValueError("Invalid state format")

        if "max_laps" in state:
            self.max_laps = state["max_laps"]

        # Clear current timers
        self._timers.clear()

        # Restore each timer
        for name, timer_data in state["timers"].items():
            timer = Timer.from_dict(timer_data)

            # Handle timers that were running when saved
            if resume_running and timer_data.get("_was_running", False):
                # Resume the timer
                timer.start_time = current_time
                # No need to adjust total_elapsed - it already includes time up to save point

            self._timers[name] = timer

        # Ensure global timer exists
        if self.GLOBAL_TIMER_NAME not in self._timers:
            self._timers[self.GLOBAL_TIMER_NAME] = self._create_timer(self.GLOBAL_TIMER_NAME)
