
import numpy as np
from rich.table import Table
from omegaconf import OmegaConf

from .dashboard import DashboardComponent
from .dashboard import c1, b1, c2, b2, c3
import wandb

class WanDb(DashboardComponent):
    def __init__(self, wandb_cfg: OmegaConf):
        super().__init__()
        self.wandb_cfg = wandb_cfg

    def render(self):
        table = Table(box=None, expand=False, pad_edge=False)

        if self.wandb_cfg.enabled:
            wandb_status = "Connecting"
        else:
            wandb_status = "Disabled"

        if wandb.run:
            wandb_status = "(not tracking)" if not self.wandb_cfg.track else ""
            wandb_status += f" {wandb.run.get_url()}"
        table.add_row(f' {c1}WandDb: {b2}{wandb_status}')
        return table
