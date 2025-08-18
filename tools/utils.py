import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbRun
from metta.core.distributed_config import DistributedConfig
from metta.mettagrid import MettaGridEnv
from metta.rl.policy_initializer import PolicyInitializer
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig


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


def get_policy_initializer_from_cfg(
    agent_cfg: DictConfig,
    system_cfg: SystemConfig,
    trainer_cfg: TrainerConfig,
    metta_grid_env: MettaGridEnv,
    distributed_config: DistributedConfig,
    device: torch.device,
) -> PolicyInitializer:
    """Create PolicyInitializer from config objects."""
    return PolicyInitializer(
        agent_cfg=agent_cfg,
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        metta_grid_env=metta_grid_env,
        distributed_config=distributed_config,
        device=device,
    )
