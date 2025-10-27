"""Stability regularization for latent representations."""

from __future__ import annotations

from typing import Any, Sequence

import torch
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

        segments, horizon = map(int, minibatch.batch_size)
        latent = self._reshape_latent(latent.to(dtype=torch.float32), segments, horizon)

        if horizon < 2:
            return self._record_zero(shared_loss_data)

        deltas = latent.diff(dim=1)
        mask = self._valid_transition_mask(minibatch, segments, horizon)
        denom, squared_deltas = self._masked_squared_deltas(deltas, mask)
        if denom <= self.loss_cfg.epsilon:
            return self._record_zero(shared_loss_data)

        loss = squared_deltas.sum() / denom
        loss = loss * self.loss_cfg.loss_coef

        mean_delta = self._mean_delta_norm(deltas, mask)
        self._record_loss(loss, mean_delta)

        return loss, shared_loss_data, False

    def _reshape_latent(self, latent: Tensor, segments: int, horizon: int) -> Tensor:
        if latent.dim() == 2:
            return latent.view(segments, horizon, -1)
        if latent.dim() == 3 and latent.shape[:2] == (segments, horizon):
            return latent
        raise ValueError(
            "Latent tensor must align with minibatch dimensions; "
            f"expected ({segments}, {horizon}, *), received {tuple(latent.shape)}."
        )

    def _valid_transition_mask(self, minibatch: TensorDict, segments: int, horizon: int) -> Tensor | None:
        if not self.loss_cfg.exclude_done_transitions:
            return None

        mask: Tensor | None = None
        for key in ("dones", "truncateds"):
            flags = minibatch.get(key)
            if flags is None:
                continue
            flags = flags.to(torch.bool)
            if flags.dim() == 3:
                flags = flags.squeeze(-1)
            if flags.shape != (segments, horizon):
                raise ValueError(f"Expected '{key}' tensor to have shape ({segments}, {horizon}[, 1]).")
            current = ~flags[:, :-1]
            mask = current if mask is None else mask & current
        return mask

    def _masked_squared_deltas(self, deltas: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        feature_dim = deltas.shape[-1]
        if mask is not None:
            mask = mask.to(device=deltas.device, dtype=deltas.dtype)
            # Broadcast mask to feature dimension so we keep per-feature scaling.
            deltas = deltas * mask.unsqueeze(-1)
            valid_transitions = mask.sum()
        else:
            valid_transitions = torch.tensor(
                deltas.shape[0] * deltas.shape[1], device=deltas.device, dtype=deltas.dtype
            )
        denom = valid_transitions * feature_dim
        return denom, deltas.square()

    def _mean_delta_norm(self, deltas: Tensor, mask: Tensor | None) -> float:
        with torch.no_grad():
            norm = deltas.norm(dim=-1)
            if mask is None:
                return float(norm.mean().item())
            mask = mask.to(device=deltas.device, dtype=norm.dtype)
            denom = mask.sum()
            if denom <= self.loss_cfg.epsilon:
                return 0.0
            return float((norm * mask).sum().item() / denom.item())

    def _record_loss(self, loss: Tensor, mean_delta: float) -> None:
        self.loss_tracker["stable_latent_loss"].append(float(loss.item()))
        self.loss_tracker["stable_latent_delta_l2"].append(mean_delta)

    def _record_zero(self, shared_loss_data: TensorDict) -> tuple[Tensor, TensorDict, bool]:
        zero_loss = self._zero()
        self.loss_tracker["stable_latent_loss"].append(float(zero_loss.item()))
        self.loss_tracker["stable_latent_delta_l2"].append(0.0)
        return zero_loss, shared_loss_data, False
