"""
This file implements a Profile class for tracking and measuring performance metrics during training.

The Profile class provides:
- Tracking of steps per second (SPS)
- Timing of different training phases (eval, env, forward passes, etc)
- Uptime and remaining time estimation
- Profilers for detailed timing breakdowns

Key features:
- Tracks overall training progress and timing
- Measures performance of different training components
- Provides timing breakdowns for analysis
- Helps monitor training efficiency
"""

import time


def _fmt_perf(time: float, uptime: float) -> float:
    return 100 * (time / uptime if uptime > 0 else 0)


class Profiler:
    """Simple profiler class to replace pufferlib.utils.Profiler"""
    def __init__(self, elapsed=True, calls=True, memory=False, pytorch_memory=False):
        self.elapsed = 0.0 if elapsed else None
        self.calls = 0 if calls else None
        self.memory = None
        self.pytorch_memory = None
        self.prev = 0
        
        self.track_elapsed = elapsed
        self.track_calls = calls
        self.track_memory = memory
        self.track_pytorch_memory = pytorch_memory
        
        self._start_time = None

    @property
    def serial(self):
        return {
            'elapsed': self.elapsed,
            'calls': self.calls,
            'memory': self.memory,
            'pytorch_memory': self.pytorch_memory,
            'delta': self.delta
        }

    @property
    def delta(self):
        ret = self.elapsed - self.prev if self.elapsed is not None else None
        self.prev = self.elapsed if self.elapsed is not None else 0
        return ret

    def __enter__(self):
        if self.track_elapsed:
            self._start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.track_elapsed and self._start_time is not None:
            self.elapsed += time.time() - self._start_time
            self._start_time = None
        if self.track_calls:
            self.calls += 1

    def __repr__(self):
        parts = []
        if self.track_elapsed and self.elapsed is not None:
            parts.append(f'Elapsed: {self.elapsed:.4f} s')
        if self.track_calls and self.calls is not None:
            parts.append(f'Calls: {self.calls}')
        return ", ".join(parts) if parts else "Profiler()"

    # Aliases for use without context manager
    start = __enter__
    stop = __exit__


def profile(func):
    """Decorator to automatically profile method execution time."""
    name = func.__name__

    def wrapper(*args, **kwargs):
        self = args[0]

        if not hasattr(self, '_timers'):
            self._timers = {}

        if name not in self._timers:
            self._timers[name] = Profiler()

        timer = self._timers[name]

        with timer:
            result = func(*args, **kwargs)

        return result

    return wrapper


class Profile:
    SPS: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0

    def __init__(self):
        self.start = time.time()
        self.env = Profiler()
        self.eval_forward = Profiler()
        self.eval_misc = Profiler()
        self.train_forward = Profiler()
        self.learn = Profiler()
        self.train_misc = Profiler()
        self.prev_steps = 0

    def __iter__(self):
        yield "SPS", self.SPS
        yield "uptime", self.uptime
        yield "remaining", self.remaining
        yield "eval_time", _fmt_perf(self.eval_time, self.uptime)
        yield "env_time", _fmt_perf(self.env_time, self.uptime)
        yield "eval_forward_time", _fmt_perf(self.eval_forward_time, self.uptime)
        yield "eval_misc_time", _fmt_perf(self.eval_misc_time, self.uptime)
        yield "train_time", _fmt_perf(self.train_time, self.uptime)
        yield "train_forward_time", _fmt_perf(self.train_forward_time, self.uptime)
        yield "learn_time", _fmt_perf(self.learn_time, self.uptime)
        yield "train_misc_time", _fmt_perf(self.train_misc_time, self.uptime)

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, global_step, total_timesteps, timers):
        if global_step == 0:
            return True

        uptime = time.time() - self.start

        self.SPS = (global_step - self.prev_steps) / (uptime - self.uptime)
        self.prev_steps = global_step
        self.uptime = uptime

        self.remaining = (total_timesteps - global_step) / self.SPS
        
        # Handle case where timers might be empty or missing expected keys
        if timers and "_rollout" in timers:
            self.eval_time = timers["_rollout"].elapsed
        if timers and "_train" in timers:
            self.train_time = timers["_train"].elapsed
            
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True
