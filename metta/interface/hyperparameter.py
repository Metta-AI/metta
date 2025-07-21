import logging
from typing import Optional

import torch
from omegaconf import DictConfig

from metta.rl.hyperparameter_scheduler import HyperparameterScheduler as BaseHyperparameterScheduler
from metta.rl.trainer_config import HyperparameterSchedulerConfig, PPOConfig

__all__ = ["SimpleHyperparameterScheduler"]


class SimpleHyperparameterScheduler:
    """Lightweight wrapper around the RL HyperparameterScheduler using simple kwargs."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_timesteps: int,
        learning_rate: Optional[float] = None,
        ppo_config: Optional[PPOConfig] = None,
        scheduler_config: Optional[HyperparameterSchedulerConfig] = None,
    ) -> None:
        # Default args
        learning_rate = learning_rate or optimizer.param_groups[0]["lr"]
        ppo_config = ppo_config or PPOConfig()
        scheduler_config = scheduler_config or HyperparameterSchedulerConfig()

        # Build DictConfig expected by BaseHyperparameterScheduler
        ppo_dict = {
            "ppo_clip_coef": ppo_config.clip_coef,
            "ppo_ent_coef": ppo_config.ent_coef,
            "ppo_vf_clip_coef": ppo_config.vf_clip_coef,
            "ppo_l2_reg_loss_coef": ppo_config.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": ppo_config.l2_init_loss_coef,
        }
        cfg = DictConfig(
            {
                "ppo": ppo_dict,
                "optimizer": {"learning_rate": learning_rate},
                "hyperparameter_scheduler": scheduler_config.model_dump(),
            }
        )

        self._scheduler = BaseHyperparameterScheduler(cfg, optimizer, total_timesteps, logging)

    def step(self, current_timestep: int) -> None:  # pragma: no cover
        """Advance the scheduler."""
        self._scheduler.step(current_timestep)
