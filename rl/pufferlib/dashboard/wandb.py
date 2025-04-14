import wandb
from omegaconf import OmegaConf
from rich.table import Table

from .dashboard import DashboardComponent, b2, c1

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
        table.add_row(f" {c1}WandDb: {b2}{wandb_status}")
        return table
