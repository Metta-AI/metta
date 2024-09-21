import wandb
from omegaconf import OmegaConf

def init_wandb(cfg, resume=True, name=None):
    #os.environ["WANDB_SILENT"] = "true"
    if not cfg.wandb.enabled and not cfg.wandb.track:
        return

    if wandb.run is not None:
        print("wandb.init() has already been called, ignoring.")
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
