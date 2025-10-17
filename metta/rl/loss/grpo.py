"""Guided Reward Policy Optimization loss built atop PPO infrastructure."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.ppo import PPO, PPOConfig
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config


class GRPOConfig(PPOConfig):
    """Configuration for GRPO loss."""

    ratio_smoothing_beta: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Linear blend applied to importance ratios when shaping advantages.",
    )
    gradient_penalty_coef: float = Field(
        default=0.1,
        ge=0.0,
        description="Coefficient for the variance-reducing gradient penalty on importance ratios.",
    )
    kl_penalty_coef: float = Field(
        default=0.2,
        ge=0.0,
        description="Coefficient on the KL deviation penalty.",
    )
    kl_target: float | None = Field(
        default=0.01,
        ge=0.0,
        description="Target KL divergence; if None the raw KL is penalised.",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Config,
    ) -> "GRPO":
        return GRPO(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class GRPO(PPO):
    """Guided Reward Policy Optimisation with ratio smoothing and KL penalties."""

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
    ) -> Tensor:
        cfg = self.loss_cfg
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["values"]

        importance_sampling_ratio = self._importance_ratio(new_logprob, old_logprob)

        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            cfg.gamma,
            cfg.gae_lambda,
            cfg.vtrace.rho_clip,
            cfg.vtrace.c_clip,
            self.device,
        )

        adv = normalize_advantage_distributed(adv, cfg.norm_adv)
        adv = prio_weights * adv

        smoothing = (1.0 - cfg.ratio_smoothing_beta) + cfg.ratio_smoothing_beta * importance_sampling_ratio
        guided_adv = adv * smoothing

        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
            minibatch,
            new_logprob,
            entropy,
            newvalue,
            importance_sampling_ratio,
            guided_adv,
        )

        base_loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

        gradient_penalty = (importance_sampling_ratio - 1.0).pow(2).mean()
        if cfg.kl_target is None:
            kl_penalty_term = approx_kl.pow(2)
        else:
            excess = torch.relu(approx_kl - cfg.kl_target)
            kl_penalty_term = excess.pow(2)

        total_loss = base_loss + cfg.gradient_penalty_coef * gradient_penalty + cfg.kl_penalty_coef * kl_penalty_term

        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.replay.update(indices, update_td)

        self._track("policy_loss", pg_loss)
        self._track("value_loss", v_loss)
        self._track("entropy", entropy_loss)
        self._track("approx_kl", approx_kl)
        self._track("clipfrac", clipfrac)
        self._track("importance", importance_sampling_ratio.mean())
        self._track("current_logprobs", new_logprob.mean())
        self._track("grpo_gradient_penalty", gradient_penalty)
        self._track("grpo_kl_penalty", kl_penalty_term)
        self._track("grpo_ratio_smoothing", smoothing.mean())

        return total_loss
