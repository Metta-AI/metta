from typing import Any

import einops
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext


class DynamicsConfig(LossConfig):
    returns_step_look_ahead: int = Field(default=1)
    unroll_steps: int = Field(default=0, ge=0, le=20)
    returns_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = Field(default=1.0, ge=0, le=1.0)

    @property
    def enabled(self) -> bool:
        return self.unroll_steps > 0

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
    ) -> "Dynamics":
        """Create Dynamics loss instance."""
        return Dynamics(policy, trainer_cfg, vec_env, device, instance_name, self)


class Dynamics(Loss):
    """The dynamics term in the Muesli loss."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.cfg, DynamicsConfig)

        # Get the underlying module (handle DDP wrapping)
        policy_module = getattr(self.policy, "module", self.policy)

        # Validate policy has dynamics modules if unroll_steps > 0
        if self.cfg.unroll_steps > 0:
            if not hasattr(policy_module, "returns_pred") or policy_module.returns_pred is None:
                raise RuntimeError(
                    f"Dynamics loss requires unroll_steps={self.cfg.unroll_steps} but policy "
                    "does not have dynamics modules. Set unroll_steps in the policy config "
                    "(e.g., ViTDefaultConfig(unroll_steps=5))."
                )

        # Set unroll_steps on policy (may override, but should match)
        if hasattr(policy_module, "unroll_steps"):
            policy_module.unroll_steps = self.cfg.unroll_steps

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        assert isinstance(self.cfg, DynamicsConfig)

        policy_td = shared_loss_data["policy_td"]
        assert isinstance(policy_td, TensorDict)

        returns_pred: Tensor = policy_td["returns_pred"].float()
        reward_pred: Tensor = policy_td["reward_pred"].float()

        # Reshape from (B, T, 1) to (B, T)
        returns_pred = einops.rearrange(returns_pred, "b t 1 -> b (t 1)")
        reward_pred = einops.rearrange(reward_pred, "b t 1 -> b (t 1)")

        # Targets
        sampled_mb = shared_loss_data["sampled_mb"]
        returns = sampled_mb["returns"]
        rewards = sampled_mb["rewards"]

        # Align predictions with future targets
        look_ahead = self.cfg.returns_step_look_ahead

        future_returns = returns[look_ahead:]
        returns_pred_aligned = returns_pred[:-look_ahead]
        returns_loss = F.mse_loss(returns_pred_aligned, future_returns) * self.cfg.returns_pred_coef

        future_rewards = rewards[1:]
        reward_pred_aligned = reward_pred[:-1]
        reward_loss = F.mse_loss(reward_pred_aligned, future_rewards) * self.cfg.reward_pred_coef

        # K-step unrolling losses (computed by policy.forward())
        K = self.cfg.unroll_steps
        if K > 0:
            # Read unrolled predictions from TensorDict (written by policy.forward())
            # Shape: (B, T, K) - already padded to full T, first T_eff positions are valid
            unrolled_rewards = policy_td["unrolled_rewards"]  # (B, T, K)
            unrolled_returns = policy_td["unrolled_returns"]  # (B, T, K)

            B, T = shared_loss_data.batch_size
            T_eff = T - K

            total_unroll_reward_loss = torch.tensor(0.0, device=self.device)
            total_unroll_returns_loss = torch.tensor(0.0, device=self.device)

            for k in range(K):
                # Target rewards/returns at t+k+1
                target_r = rewards[:, k + 1 : k + 1 + T_eff]
                target_v = returns[:, k + 1 : k + 1 + T_eff]

                # Predictions are at positions 0:T_eff for unroll step k
                r_pred_k = unrolled_rewards[:, :T_eff, k]
                v_pred_k = unrolled_returns[:, :T_eff, k]

                total_unroll_reward_loss += F.mse_loss(r_pred_k, target_r)
                total_unroll_returns_loss += F.mse_loss(v_pred_k, target_v)

            reward_loss += (total_unroll_reward_loss / K) * self.cfg.reward_pred_coef
            returns_loss += (total_unroll_returns_loss / K) * self.cfg.returns_pred_coef

            # Pass unrolled logits to MuesliModel loss (already (B, T, K, A))
            shared_loss_data["muesli_unrolled_logits"] = policy_td["unrolled_logits"]

        self.loss_tracker["dynamics_returns_loss"].append(float(returns_loss.item()))
        self.loss_tracker["dynamics_reward_loss"].append(float(reward_loss.item()))

        return returns_loss + reward_loss, shared_loss_data, False
