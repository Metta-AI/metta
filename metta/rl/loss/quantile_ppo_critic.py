from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class QuantilePPOCriticConfig(LossConfig):
    vf_clip_coef: float = Field(default=0.1, ge=0)
    vf_coef: float = Field(default=0.49657103419303894, ge=0)
    # Value loss clipping toggle
    clip_vloss: bool = True

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "QuantilePPOCritic":
        return QuantilePPOCritic(policy, trainer_cfg, env, device, instance_name, self)


class QuantilePPOCritic(Loss):
    """Quantile PPO value loss."""

    __slots__ = (
        "advantages",
        "burn_in_steps",
        "burn_in_steps_iter",
        "num_quantiles",
        "tau_hat",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: "QuantilePPOCriticConfig",
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        else:
            self.burn_in_steps = 0
        self.burn_in_steps_iter = 0

        if not hasattr(self.policy, "critic_quantiles"):
            raise ValueError("Policy must expose 'critic_quantiles' attribute for QuantilePPOCritic")
        self.num_quantiles = self.policy.critic_quantiles

        # Pre-compute tau_hat (quantile midpoints)
        # Shape: [1, N]
        i = torch.arange(self.num_quantiles, device=self.device, dtype=torch.float32)
        self.tau_hat = ((2 * i + 1) / (2 * self.num_quantiles)).view(1, -1)

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        # Values are now [N] instead of scalar
        values_spec = UnboundedContinuous(shape=torch.Size([self.num_quantiles]), dtype=torch.float32)

        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            values=values_spec,
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Rollout step: forward policy and store experience with optional burn-in."""
        with torch.no_grad():
            if "actions" in td.keys():
                self.policy.forward(td, action=td["actions"])
            else:
                self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        env_slice = self._training_env_id(context)
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"values"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        minibatch = shared_loss_data["sampled_mb"]
        indices = shared_loss_data["indices"][:, 0]
        # compute advantages on the first mb
        if mb_idx == 0:
            # Calculate mean values for GAE
            values_quantiles = self.replay.buffer["values"]  # [T, B, N]
            values_mean = values_quantiles.mean(dim=-1)  # [T, B]

            centered_rewards = self.replay.buffer["rewards"] - self.replay.buffer["reward_baseline"]
            self.advantages = compute_advantage(
                values_mean,
                centered_rewards,
                self.replay.buffer["dones"],
                torch.ones_like(values_mean),
                torch.zeros_like(values_mean, device=self.device),
                self.trainer_cfg.advantage.gamma,
                self.trainer_cfg.advantage.gae_lambda,
                self.device,
                1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
                1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
            )

        if minibatch.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

        advantages = shared_loss_data["advantages_full"][indices]

        # compute value loss
        old_values_quantiles = minibatch["values"]  # [B, N]
        old_values_mean = old_values_quantiles.mean(dim=-1)  # [B]

        # Target return is scalar
        returns = advantages + old_values_mean
        minibatch["returns"] = returns

        policy_td = shared_loss_data.get("policy_td", None)
        newvalue = None
        if policy_td is not None:
            newvalue = policy_td["values"]  # [B, N]

        if newvalue is not None:
            # Quantile Regression Loss
            # Target is 'returns' broadcasted
            target = returns.unsqueeze(-1)  # [B, 1]

            # Calculate diffs
            # newvalue: [B, N]
            # target: [B, 1]
            # We want diff between target and each quantile
            # diff = target - newvalue

            if self.cfg.clip_vloss:
                # Clip based on change from old_values_quantiles
                vf_clip_coef = self.cfg.vf_clip_coef
                # We clip the quantiles themselves
                newvalue_clipped = old_values_quantiles + torch.clamp(
                    newvalue - old_values_quantiles,
                    -vf_clip_coef,
                    vf_clip_coef,
                )

                # Loss with unclipped
                loss_unclipped = self.quantile_loss(newvalue, target)

                # Loss with clipped
                loss_clipped = self.quantile_loss(newvalue_clipped, target)

                v_loss = torch.max(loss_unclipped, loss_clipped).mean()
            else:
                v_loss = self.quantile_loss(newvalue, target).mean()

            # Update values in experience buffer
            update_td = TensorDict(
                {
                    "values": newvalue.view(minibatch["values"].shape).detach(),
                },
                batch_size=minibatch.batch_size,
            )
            self.replay.update(indices, update_td)
        else:
            # Fallback if no policy forward output is available.
            v_loss = self.quantile_loss(old_values_quantiles, returns.unsqueeze(-1)).mean()

        # Scale value loss by coefficient
        v_loss = v_loss * self.cfg.vf_coef
        self.loss_tracker["value_loss"].append(float(v_loss.item()))

        return v_loss, shared_loss_data, False

    def quantile_loss(self, current_quantiles: Tensor, target: Tensor) -> Tensor:
        """
        Compute quantile regression loss.
        current_quantiles: [B, N]
        target: [B, 1]
        """
        # Huber loss on difference
        # diff: [B, N]
        diff = target - current_quantiles

        # Huber loss (smooth L1)
        # beta = 1.0 (standard) or maybe adjustable?
        # PyTorch smooth_l1_loss default beta=1.0
        huber_loss = F.smooth_l1_loss(
            current_quantiles, target.expand_as(current_quantiles), reduction="none", beta=1.0
        )

        # Quantile weight: |tau - I(diff < 0)|
        # tau_hat: [1, N]
        # I(diff < 0): [B, N]
        indicator = (diff < 0).float()
        quantile_weight = torch.abs(self.tau_hat - indicator)

        # Loss = sum over quantiles of (weight * huber_loss)
        # We return mean over batch later, here just sum over quantiles
        loss = (quantile_weight * huber_loss).sum(dim=-1)

        return loss

    def on_train_phase_end(self, context: ComponentContext) -> None:
        """Compute value-function explained variance for logging."""
        with torch.no_grad():
            # Use mean of quantiles for explained variance
            values_quantiles = self.replay.buffer["values"]
            values_mean = values_quantiles.mean(dim=-1).flatten()

            y_pred = values_mean
            y_true = self.advantages.flatten() + values_mean
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

        super().on_train_phase_end(context)
