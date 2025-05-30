"""
Thin wrapper around PufferLib's Profile for cleaner integration.
"""

import time
import torch
from pufferlib.pufferl import Profile as PufferProfile


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


class Profile:
    """Thin wrapper around PufferLib's Profile with a cleaner interface."""

    def __init__(self, frequency=1):
        self._profile = PufferProfile(frequency=frequency)
        self.start_time = time.time()
        self.epoch = 0

        # Timing stats
        self.SPS = 0
        self.uptime = 0
        self.remaining = 0

    def __getattr__(self, name):
        # Create context managers for profile sections
        if name in ["env", "eval_forward", "eval_misc", "train_forward", "learn", "train_misc"]:
            return ProfileTimer(self._profile, name, self.epoch)
        # Direct access to timing data
        return self._profile.profiles.get(name, {}).get("elapsed", 0)

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
            self.SPS = global_step / elapsed
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

        # Add performance percentages
        for k, v in self.get_performance_stats().items():
            yield k, v
