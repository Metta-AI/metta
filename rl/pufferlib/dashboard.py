import time
from collections import defaultdict
from threading import Thread
from collections import deque

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
    def __init__(self, cfg: OmegaConf, clear=False, delay=1, max_stats=5, max_msg_log=10):
        super().__init__()
        self.cfg = cfg
        self.utilization = Utilization(delay=10)
        self.global_step = 0
        self.epoch = 0
        self.profile = Profile()
        self.losses = {}
        self.stats = defaultdict(list)
        self.msg_log = deque(maxlen=max_msg_log)
        self.policy_params = 0
        self.clear = clear
        self.max_stats = max_stats
        self.checkpoint = {
            "saved_at": 0,
            "path": "",
            "steps": 0,
            "epoch": 0,
        }
        self.wandb_model = {
            "saved_at": 0,
            "name": "",
            "steps": 0,
            "epoch": 0,
        }
        self.carbs = {
            "num_observations": 0,
            "num_suggestions": 0,
            "last_metric": 0,
            "last_run_time": 0,
            "last_run_success": False,
            "num_failures": 0,
        }

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.print()
            time.sleep(self.delay)

    def stop(self):
        self.utilization.stop()
        self.stopped = True

    def update_checkpoint(self, path, steps, epoch):
        self.checkpoint["path"] = path
        self.checkpoint["steps"] = steps
        self.checkpoint["epoch"] = epoch
        self.checkpoint["saved_at"] = time.time()

    def update_wandb_model(self, artifact):
        self.wandb_model["name"] = artifact.name
        self.wandb_model["epoch"] = artifact.metadata["epoch"]
        self.wandb_model["steps"] = artifact.metadata["agent_step"]
        self.wandb_model["saved_at"] = time.time()

    def update_carbs(self, num_observations, num_suggestions, last_metric, last_run_time, last_run_success, num_failures):
        self.carbs["num_observations"] = num_observations
        self.carbs["num_suggestions"] = num_suggestions
        self.carbs["last_metric"] = last_metric
        self.carbs["last_run_time"] = last_run_time
        self.carbs["last_run_success"] = last_run_success
        self.carbs["num_failures"] = num_failures

    def set_policy(self, policy):
        self.policy_params = count_params(policy)

    def log(self, msg):
        self.msg_log.append(msg)

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

        l = Table(box=None, expand=True)
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
            if i >= self.max_stats:
                break
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
            i += 1

        for i in range(self.max_stats - i):
            u = left if i % 2 == 0 else right
            u.add_row('', '')

        table = Table(box=None, expand=False, pad_edge=False)
        dashboard.add_row(table)
        table.add_row(f' {c1}Checkpoint: {b2}{self.checkpoint["path"]} epoch: {self.checkpoint["epoch"]} steps: {self.checkpoint["steps"]}')
        table.add_row(f' {c1}WandDb Model: {b2}{self.wandb_model["name"]} epoch: {self.wandb_model["epoch"]} steps: {self.wandb_model["steps"]}')
        wandb_status = "Disabled"
        if wandb.run:
            wandb_status = f"(e) ({wandb.run.name}): {wandb.run.url}"
            if self.cfg.wandb.track:
                wandb_status = f"(t) ({wandb.run.name}): {wandb.run.url}"
        else:
            if self.cfg.wandb.track:
                wandb_status = "(t): Not Initialized"
        table.add_row(f' {c1}WandDb: {b2}{wandb_status}')

        table.add_row(
            f' {c1}Carbs: {b2}' +
            f'o: {self.carbs["num_observations"]} ' +
            f's: {self.carbs["num_suggestions"]} ' +
            f'm: {self.carbs["last_metric"]} ' +
            f't: {self.carbs["last_run_time"]} ' +
            f't: {self.carbs["last_run_success"]} ' +
            f'f: {self.carbs["num_failures"]}')

        table = Table(box=ROUND_OPEN, expand=True, pad_edge=False)
        dashboard.add_row(table)
        for msg in self.msg_log:
            table.add_row(f'{c2}{msg}')
        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())


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
