
import numpy as np
from rich.table import Table

from rl.pufferlib.utilization import Utilization

from .component import DashboardComponent
from .component import c1, b1, c2, b2, c3
from .component import abbreviate, duration, fmt_perf
from rl.pufferlib.policy import count_params


class Policy(DashboardComponent):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.num_params = 0
        self.saved_at = 0
        self.epoch = 0
        self.agent_steps = 0
        self.wandb_model_name = None
        self.wandb_url = None

    def set_policy(self, policy):
        self.num_params = count_params(policy)

    def render(self):
        table = Table(box=None, expand=True, pad_edge=False)
        table.add_row(f'{c2}Policy Params', abbreviate(self.num_params))
        table.add_row(f'{c2}Epoch', abbreviate(self.epoch))
        table.add_row(f'{c2}Agent Steps', abbreviate(self.agent_steps))
        table.add_row(f'{c2}Wandb Model', abbreviate(self.wandb_model_name))
        table.add_row(f' {c1}Checkpoint: {b2}{self.checkpoint["path"]} epoch: {self.checkpoint["epoch"]} steps: {self.checkpoint["steps"]}')
        table.add_row(f' {c1}WandDb Model: {b2}{self.wandb_model["name"]} epoch: {self.wandb_model["epoch"]} steps: {self.wandb_model["steps"]}')
        return table
