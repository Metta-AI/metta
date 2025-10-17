"""Mirror-style policy optimization loss built on PPO infrastructure."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.ppo import PPO, PPOConfig
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config


class CMPOConfig(PPOConfig):
    """Configuration for the CMPO loss.

    CMPO (Clip Mirror Policy Optimisation) augments PPO with an explicit penalty on
    policy updates that drift too far from the behaviour policy. We model this by
    introducing a configurable mirror penalty on log-prob deviations.
    """

    mirror_coef: float = Field(
        default=0.5,
        ge=0.0,
        description="Strength of the mirror penalty that regularises policy updates.",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Config,
    ) -> "CMPO":
        """Instantiate a CMPO loss module."""
        return CMPO(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class CMPO(PPO):
    """A lightweight CMPO variant implemented as a PPO extension."""

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
    ) -> Tensor:
        """Apply PPO update with an additional mirror penalty."""
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        mirror_penalty = (new_logprob - old_logprob).pow(2).mean()

        base_loss = super()._process_minibatch_update(minibatch, policy_td, indices, prio_weights)

        total_loss = base_loss + mirror_penalty * self.loss_cfg.mirror_coef
        self._track("cmpo_mirror_penalty", mirror_penalty)
        return total_loss
