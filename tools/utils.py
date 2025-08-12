import multiprocessing

import torch
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


def calculate_default_num_workers(is_serial: bool) -> int:
    if is_serial:
        return 1

    cpu_count = multiprocessing.cpu_count() or 1

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    ideal_workers = (cpu_count // 2) // num_gpus

    # Round down to nearest power of 2
    num_workers = 1
    while num_workers * 2 <= ideal_workers:
        num_workers *= 2

    return max(1, num_workers)
