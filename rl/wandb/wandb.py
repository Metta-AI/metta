import wandb
import os
from omegaconf import OmegaConf

def init_wandb(cfg, resume=True, name=None):
    if not cfg.wandb.enabled:
        assert not cfg.wandb.track, "wandb.track wont work if wandb.enabled is False"
        return

    wandb.init(
        id=cfg.experiment or wandb.util.generate_id(),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.wandb.group,
        allow_val_change=True,
        name=name or cfg.wandb.name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )

    wandb.save(os.path.join(cfg.data_dir, cfg.experiment, "*.log"), policy="live")
    wandb.save(os.path.join(cfg.data_dir, cfg.experiment, "*.yaml"), policy="live")
