import functools
import time
from typing import Literal, overload

from pufferlib.pufferl import Profile as PufferProfile


def _fmt_perf(elapsed, total):
    """Format performance metric as percentage of total time."""
    pct = 100 * elapsed / total if total > 0 else 0
    return pct


class ProfileTimer:
    """Simple context manager that wraps PufferLib's profiling calls."""

    def __init__(self, puffer_profile, name, epoch):
        self.puffer_profile = puffer_profile
        self.name = name
        self.epoch = epoch

    def __enter__(self):
        self.puffer_profile(self.name, self.epoch, nest=False)
        return self

    def __exit__(self, *args):
        pass  # PufferLib automatically ends when next section starts


def profile_section(section_name):
    """Decorator to profile a section of code using the Profile context manager."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.profile.start_epoch(self.epoch, section_name)
            try:
                result = func(self, *args, **kwargs)
            finally:
                self.profile.end_epoch()
            return result

        return wrapper

    return decorator


class Profile:
    """Thin wrapper around PufferLib's Profile with a cleaner interface."""

    def __init__(self, frequency=1):
        self._profile = PufferProfile(frequency=frequency)
        self.start_time = time.time()
        self.start_agent_steps = 0
        self.epoch = 0

        # Timing stats
        self.SPS = 0
        self.uptime = 0
        self.remaining = 0

    # Type overloads for context managers
    @overload
    def __getattr__(
        self, name: Literal["env", "eval_forward", "eval_misc", "train_forward", "learn", "train_misc"]
    ) -> ProfileTimer: ...

    # Type overload for timing data
    @overload
    def __getattr__(self, name: str) -> float: ...

    def __getattr__(self, name):
        # Create context managers for profile sections
        if name in ["env", "eval_forward", "eval_misc", "train_forward", "learn", "train_misc"]:
            return ProfileTimer(self._profile, name, self.epoch)

        # Direct access to timing data for individual metrics
        if name in [
            "eval_time",
            "train_time",
            "env_time",
            "eval_forward_time",
            "eval_misc_time",
            "train_forward_time",
            "learn_time",
            "train_misc_time",
        ]:
            # Remove '_time' suffix to get the profile key
            profile_key = name.replace("_time", "")
            return self._profile.profiles.get(profile_key, {}).get("elapsed", 0)

        # Default behavior
        return self._profile.profiles.get(name, {}).get("elapsed", 0)

    @property
    def epoch_time(self) -> float:
        """Total time for the epoch (train + eval)."""
        return self.train_time + self.eval_time

    def start_epoch(self, epoch, section="eval"):
        """Start profiling for a new epoch."""
        self.epoch = epoch
        self._profile(section, epoch)

    def end_epoch(self):
        """End profiling for the current epoch."""
        self._profile.end()

    def update_stats(self, global_step, total_timesteps):
        """Update SPS and timing statistics."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        if elapsed > self.uptime:  # Avoid division by zero on first call
            self.SPS = (global_step - self.start_agent_steps) / elapsed
            self.remaining = (total_timesteps - global_step) / self.SPS if self.SPS > 0 else 0

        self.uptime = elapsed

        # Clear buffers for next iteration
        self._profile.clear()

    def get_performance_stats(self):
        """Get performance statistics for logging."""
        stats = {}
        profiles = self._profile.profiles

        for name in ["env", "eval_forward", "eval_misc", "train_forward", "learn", "train_misc"]:
            elapsed = profiles.get(name, {}).get("elapsed", 0)
            stats[f"{name}_time"] = 100 * elapsed / self.uptime if self.uptime > 0 else 0

        # Add aggregate times
        eval_time = profiles.get("eval", {}).get("elapsed", 0)
        train_time = profiles.get("train", {}).get("elapsed", 0)
        stats["eval_time"] = 100 * eval_time / self.uptime if self.uptime > 0 else 0
        stats["train_time"] = 100 * train_time / self.uptime if self.uptime > 0 else 0

        return stats

    def __iter__(self):
        """For backward compatibility with logging."""
        yield "SPS", self.SPS
        yield "uptime", self.uptime
        yield "remaining", self.remaining

        # Compute aggregate times as sum of components
        # This is more accurate than using the decorator-based measurements
        eval_time_computed = self.eval_misc_time + self.env_time + self.eval_forward_time
        train_time_computed = self.train_misc_time + self.train_forward_time + self.learn_time

        # Yield formatted performance metrics
        yield "eval_time", _fmt_perf(eval_time_computed, self.uptime)
        yield "env_time", _fmt_perf(self.env_time, self.uptime)
        yield "eval_forward_time", _fmt_perf(self.eval_forward_time, self.uptime)
        yield "eval_misc_time", _fmt_perf(self.eval_misc_time, self.uptime)
        yield "train_time", _fmt_perf(train_time_computed, self.uptime)
        yield "train_forward_time", _fmt_perf(self.train_forward_time, self.uptime)
        yield "learn_time", _fmt_perf(self.learn_time, self.uptime)
        yield "train_misc_time", _fmt_perf(self.train_misc_time, self.uptime)
