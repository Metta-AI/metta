from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbRun


# TODO: We should aim to remove this once hydra is fully out of our systems
def get_policy_store_from_cfg(cfg: DictConfig, wandb_run: WandbRun | None = None) -> PolicyStore:
    policy_store = PolicyStore(
        device=cfg.device,
        wandb_run=wandb_run,
        data_dir=getattr(cfg, "data_dir", None),
        wandb_entity=cfg.wandb.entity if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "entity") else None,
        wandb_project=cfg.wandb.project if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "project") else None,
        pytorch_cfg=getattr(cfg, "pytorch", None),
    )
    return policy_store
