import wandb
import os
import socket
from omegaconf import OmegaConf

# os.environ["WANDB_SILENT"] = "true"

class WandbContext:
    def __init__(self, cfg, resume=True, name=None):
        self.cfg = cfg
        self.resume = resume
        self.name = name or cfg.wandb.name
        self.run = None

    def __enter__(self):
        if not self.cfg.wandb.enabled:
            assert not self.cfg.wandb.track, "wandb.track won't work if wandb.enabled is False"
            return None

        self.run = wandb.init(
            id=self.cfg.run or wandb.util.generate_id(),
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            group=self.cfg.wandb.group,
            allow_val_change=True,
            name=self.name,
            monitor_gym=True,
            save_code=True,
            resume=self.resume,
            tags=[
                "hostname:" + os.environ.get("METTA_HOST", "unknown"),
                "user:" + os.environ.get("METTA_USER", "unknown"),
                "ip:" + socket.gethostbyname(socket.gethostname())
            ]
        )

        wandb.save(os.path.join(self.cfg.run_dir, "*.log"), base_path=self.cfg.run_dir, policy="live")
        wandb.save(os.path.join(self.cfg.run_dir, "*.yaml"), base_path=self.cfg.run_dir, policy="live")

        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            wandb.finish(quiet=True)
