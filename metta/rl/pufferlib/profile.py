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

import pufferlib
import pufferlib.pytorch
import pufferlib.utils

def _fmt_perf(time: float, uptime: float) -> float:
    return 100 * (time / uptime if uptime > 0 else 0)


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
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
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
        self.eval_time = timers["_evaluate"].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = timers["_train"].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True
