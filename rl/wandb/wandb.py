import wandb
from omegaconf import OmegaConf

def init_wandb(cfg, resume=True):
    #os.environ["WANDB_SILENT"] = "true"

    wandb.init(
        id=cfg.experiment or wandb.util.generate_id(),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.wandb.group,
        allow_val_change=True,
        name=cfg.wandb.name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb
