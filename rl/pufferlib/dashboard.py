import time
from collections import defaultdict
from threading import Thread

import numpy as np
import rich
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

import wandb
from rl.pufferlib.policy import count_params
from rl.pufferlib.utilization import Utilization
from rl.pufferlib.profile import Profile
class Dashboard(Thread):
    def __init__(self, cfg: OmegaConf, profile: Profile, clear=False, delay=1, max_stats=[0]):
        super().__init__()
        self.utilization = Utilization(delay=10)
        self.global_step = 0
        self.epoch = 0
        self.profile = profile
        self.losses = {}
        self.stats = defaultdict(list)
        self.msg = None
        self.policy_params = 0
        self.clear = clear
        self.max_stats = max_stats
        self.msg = ""
        self.wandb_status = "Disabled"
        if cfg.wandb.enabled:
            self.wandb_status = "(e: " + wandb.run.url
        if cfg.wandb.track:
            self.wandb_status = "(t): " + wandb.run.url

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.print()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


    def set_policy(self, policy):
        self.policy_params = count_params(policy)

    def log(self, msg):
        self.msg = msg

    def update_stats(self, stats):
        if len(stats) > 0:
            self.stats = stats

    def print(self):
        console = Console()
        if self.clear:
            console.clear()

        dashboard = Table(box=ROUND_OPEN, expand=True,
            show_header=False, border_style='bright_cyan')

        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)
        cpu_percent = np.mean(self.utilization.cpu_util)
        dram_percent = np.mean(self.utilization.cpu_mem)
        gpu_percent = np.mean(self.utilization.gpu_util)
        vram_percent = np.mean(self.utilization.gpu_mem)
        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)
        table.add_row(
            f':blowfish: {c1}PufferLib {b2}1.0.0',
            f'{c1}CPU: {c3}{cpu_percent:.1f}%',
            f'{c1}GPU: {c3}{gpu_percent:.1f}%',
            f'{c1}DRAM: {c3}{dram_percent:.1f}%',
            f'{c1}VRAM: {c3}{vram_percent:.1f}%',
        )

        s = Table(box=None, expand=True)
        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=16)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=8)
        s.add_row(f'{c2}Policy Params', abbreviate(self.policy_params))
        s.add_row(f'{c2}Agent Steps', abbreviate(self.global_step))
        s.add_row(f'{c2}SPS', abbreviate(self.profile.SPS))
        s.add_row(f'{c2}Epoch', abbreviate(self.epoch))
        s.add_row(f'{c2}Uptime', duration(self.profile.uptime))
        s.add_row(f'{c2}Remaining', duration(self.profile.remaining))


        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', self.profile.eval_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Forward', self.profile.eval_forward_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Env', self.profile.env_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Misc', self.profile.eval_misc_time, self.profile.uptime))
        p.add_row(*fmt_perf('Train', self.profile.train_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Forward', self.profile.train_forward_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Learn', self.profile.learn_time, self.profile.uptime))
        p.add_row(*fmt_perf('  Misc', self.profile.train_misc_time, self.profile.uptime))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0
        for metric, value in self.stats.items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
            i += 1

        for i in range(self.max_stats[0] - i):
            u = left if i % 2 == 0 else right
            u.add_row('', '')

        self.max_stats[0] = max(self.max_stats[0], i)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        table.add_row(f' {c1}WandDb {b2}{self.wandb_status}')
        table.add_row(f' {c1}Message: {c2}{self.msg}')
        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

    def close(self):
        self.utilization.stop()


ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f'{b2}{num:.0f}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}k'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}m'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}b'
    else:
        return f'{b2}{num/1e12:.1f}{c2}t'

def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100*time/uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'
