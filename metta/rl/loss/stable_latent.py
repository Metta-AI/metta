"""Stability regularization for latent representations."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment
from mettagrid.base_config import Config


class StableLatentStateConfig(Config):
    """Configuration for the stable latent state loss."""

    target_key: str | Sequence[str] = Field(
        default="core",
        description="TensorDict key that contains the latent representation to regularize.",
    )
    loss_coef: float = Field(default=1.0, ge=0.0, description="Scaling factor applied to the stability loss.")
    exclude_done_transitions: bool = Field(
        default=True,
        description="Skip transitions that cross episode boundaries when computing the penalty.",
    )
    epsilon: float = Field(default=1e-8, ge=0.0, description="Numerical stability constant for denominator checks.")

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "StableLatentStateLoss":
        """Instantiate the stable latent state loss."""
        return StableLatentStateLoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_cfg=loss_config,
        )


class StableLatentStateLoss(Loss):
    """Encourages latent representations to evolve smoothly over time."""

    __slots__ = ("_target_key",)

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_cfg: StableLatentStateConfig,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_cfg)
        self._target_key = self._normalize_key(loss_cfg.target_key)

    @staticmethod
    def _normalize_key(key: str | Sequence[str]) -> str | tuple[str, ...]:
        """Convert dotted strings into TensorDict-compatible tuple keys."""
        if isinstance(key, str):
            if "." in key:
                return tuple(part for part in key.split(".") if part)
            return key
        return tuple(key)

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        policy_td = shared_loss_data.get("policy_td")
        minibatch = shared_loss_data.get("sampled_mb")
        if policy_td is None:
            raise KeyError("StableLatentStateLoss requires 'policy_td' in shared_loss_data.")
        if minibatch is None:
            raise KeyError("StableLatentStateLoss requires 'sampled_mb' in shared_loss_data.")

        latent = policy_td.get(self._target_key)
        if latent is None:
            raise KeyError(f"StableLatentStateLoss expected key '{self._target_key}' in policy_td.")

        latent = latent.to(dtype=torch.float32)
        segments, horizon = minibatch.batch_size
        if latent.dim() == 2:
            latent = latent.view(int(segments), int(horizon), -1)
        elif latent.dim() == 3:
            if latent.shape[0] != int(segments) or latent.shape[1] != int(horizon):
                raise ValueError(
                    "Latent tensor must align with minibatch dimensions; "
                    f"expected ({int(segments)}, {int(horizon)}, *), received {tuple(latent.shape)}."
                )
        else:
            raise ValueError("Latent tensor must have shape [segments * horizon, dim] or [segments, horizon, dim].")

        if latent.shape[1] < 2:
            zero_loss = self._zero()
            self.loss_tracker["stable_latent_loss"].append(float(zero_loss.item()))
            self.loss_tracker["stable_latent_delta_l2"].append(0.0)
            return zero_loss, shared_loss_data, False

        prev_latent = latent[:, :-1, :]
        next_latent = latent[:, 1:, :]
        per_step_loss = F.mse_loss(next_latent, prev_latent, reduction="none")

        valid_mask: Tensor | None = None
        if self.loss_cfg.exclude_done_transitions:
            dones = minibatch.get("dones")
            if dones is not None:
                dones = dones.to(torch.bool)
                if dones.dim() == 3:
                    dones = dones.squeeze(-1)
                if dones.dim() != 2:
                    raise ValueError("Expected 'dones' tensor to have shape (segments, horizon[, 1]).")
                valid_mask = ~dones[:, :-1]

                truncateds = minibatch.get("truncateds")
                if truncateds is not None:
                    truncateds = truncateds.to(torch.bool)
                    if truncateds.dim() == 3:
                        truncateds = truncateds.squeeze(-1)
                    if truncateds.dim() != 2:
                        raise ValueError("Expected 'truncateds' tensor to have shape (segments, horizon[, 1]).")
                    valid_mask = torch.logical_and(valid_mask, ~truncateds[:, :-1])

        feature_dim = per_step_loss.shape[-1]

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(-1)
            per_step_loss = per_step_loss * valid_mask.to(per_step_loss.dtype)
            denom = valid_mask.sum().to(per_step_loss.dtype) * feature_dim
        else:
            denom = torch.tensor(
                per_step_loss.shape[0] * per_step_loss.shape[1] * feature_dim,
                device=per_step_loss.device,
                dtype=per_step_loss.dtype,
            )

        if denom <= self.loss_cfg.epsilon:
            zero_loss = self._zero()
            self.loss_tracker["stable_latent_loss"].append(float(zero_loss.item()))
            self.loss_tracker["stable_latent_delta_l2"].append(0.0)
            return zero_loss, shared_loss_data, False

        loss = per_step_loss.sum() / denom
        loss = loss * self.loss_cfg.loss_coef

        with torch.no_grad():
            delta = next_latent - prev_latent
            if valid_mask is not None:
                masked_delta = delta * valid_mask
                denom_delta = valid_mask.sum().to(delta.dtype)
                mean_delta = 0.0
                if denom_delta > self.loss_cfg.epsilon:
                    mean_delta = masked_delta.norm(dim=-1).sum() / denom_delta
            else:
                mean_delta = delta.norm(dim=-1).mean()

        self.loss_tracker["stable_latent_loss"].append(float(loss.item()))
        self.loss_tracker["stable_latent_delta_l2"].append(float(mean_delta))

        return loss, shared_loss_data, False
