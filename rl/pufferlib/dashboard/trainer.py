
import numpy as np
from rich.table import Table

from .dashboard import DashboardComponent
from .dashboard import c1, b1, c2, b2, c3, ROUND_OPEN
from .dashboard import abbreviate, duration, fmt_perf
from .user_stats import UserStats
from rl.pufferlib.train import PufferTrainer

class Trainer(DashboardComponent):
    def __init__(self, trainer: PufferTrainer):
        super().__init__()
        self.trainer = trainer

    def render(self):
        trainer = self.trainer
        profile = trainer.profile

        s = Table(box=ROUND_OPEN, expand=True)
        s.add_column(f"{c1}Trainer", justify='left', vertical='top')
        s.add_column(f"{c1}Value", justify='right', vertical='top')
        s.add_row(f'{c2}Agent Steps', abbreviate(trainer.global_step))
        s.add_row(f'{c2}SPS', abbreviate(profile.SPS))
        s.add_row(f'{c2}Epoch', abbreviate(trainer.epoch))
        s.add_row(f'{c2}Uptime', duration(profile.uptime))
        s.add_row(f'{c2}Remaining', duration(profile.remaining))

        p = Table(box=ROUND_OPEN, expand=True)
        p.add_column(f"{c1}Performance", justify="left")
        p.add_column(f"{c1}Time", justify="right")
        p.add_column(f"{c1}%", justify="right")
        p.add_row(*fmt_perf('Evaluate', profile.eval_time, profile.uptime))
        p.add_row(*fmt_perf('  Forward', profile.eval_forward_time, profile.uptime))
        p.add_row(*fmt_perf('  Env', profile.env_time, profile.uptime))
        p.add_row(*fmt_perf('  Misc', profile.eval_misc_time, profile.uptime))
        p.add_row(*fmt_perf('Train', profile.train_time, profile.uptime))
        p.add_row(*fmt_perf('  Forward', profile.train_forward_time, profile.uptime))
        p.add_row(*fmt_perf('  Learn', profile.learn_time, profile.uptime))
        p.add_row(*fmt_perf('  Misc', profile.train_misc_time, profile.uptime))

        l = Table(box=ROUND_OPEN, expand=True)
        l.add_column(f'{c1}Losses', justify="left",)
        l.add_column(f'{c1}Value', justify="right")
        for metric, value in self.trainer.losses.items():
            l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

        us = UserStats(self.trainer.recent_stats, max_stats=8)
        t = Table(box=None, expand=True)
        t.add_row(s, p, l, us.render())
        return t
