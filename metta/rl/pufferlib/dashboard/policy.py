from rich.table import Table

from metta.rl.pufferlib.trainer import PolicyCheckpoint

from .dashboard import ROUND_OPEN, DashboardComponent, abbreviate, c1, c2

class Policy(DashboardComponent):
    def __init__(self, checkpoint: PolicyCheckpoint):
        super().__init__()
        self.checkpoint = checkpoint
        self.wandb_model_name = None
        self.wandb_url = None

    def render(self):
        table = Table(box=ROUND_OPEN, expand=False)
        table.add_column(f"{c1}Model", justify="left", vertical="top")
        table.add_column(f"{c1}Value", justify="right", vertical="top")
        table.add_row(f"{c2}Params", abbreviate(self.checkpoint.num_params))
        if self.checkpoint.model_name:
            table.add_row(f"{c2}Epoch", abbreviate(self.checkpoint.epoch))
            table.add_row(f"{c2}Agent Steps", abbreviate(self.checkpoint.agent_step))
            table.add_row(f"{c2}Path", self.checkpoint.model_path)

        if self.checkpoint.wandb_model_artifact:
            entity = self.checkpoint.wandb_model_artifact.entity
            project = self.checkpoint.wandb_model_artifact.project
            uri = f"wandb://{self.checkpoint.wandb_model_artifact.name}"
            name, version = self.checkpoint.wandb_model_artifact.name.split(":")

            url = f"https://wandb.ai/{entity}/{project}/artifacts/model/{name}/{version}"
            table.add_row(f"{c2}Wandb Model", uri)
            table.add_row(f"{c2}Wandb URL", url)

        return table
