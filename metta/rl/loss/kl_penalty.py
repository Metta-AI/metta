from typing import Any, Tuple

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss
from metta.rl.training.component_context import ComponentContext
from metta.rl.training.training_environment import TrainingEnvironment
from mettagrid.config import Config


class KLPenaltyConfig(Config):
    """Configuration for KL divergence penalty loss."""

    # KL penalty coefficient
    kl_penalty_coef: float = Field(default=0.01, ge=0)
    # Clamp log ratio to prevent numerical issues
    max_log_ratio: float = Field(default=10.0, gt=0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Points to the KLPenalty class for initialization."""
        return KLPenalty(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class KLPenalty(Loss):
    """Standalone KL divergence penalty loss that works with any primary RL loss."""

    def get_experience_spec(self) -> Composite:
        """No additional experience storage needed - reuses data from other losses."""
        return Composite()

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """No rollout action needed - this loss operates on shared data."""
        return

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute KL penalty loss using shared policy outputs."""

        # Check if we have the required shared data
        if "policy_td" not in shared_loss_data:
            # No policy data available yet - return zero loss
            return self._zero(), shared_loss_data, False

        if "sampled_mb" not in shared_loss_data:
            # No minibatch data available yet - return zero loss
            return self._zero(), shared_loss_data, False

        # Get policy outputs and minibatch data computed by PPO
        policy_td = shared_loss_data["policy_td"]
        minibatch = shared_loss_data["sampled_mb"]

        # Extract log probabilities
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)

        # Compute KL divergence penalty
        kl_penalty_loss = self._compute_kl_penalty(new_logprob, old_logprob)

        # Scale by coefficient
        scaled_loss = self.loss_cfg.kl_penalty_coef * kl_penalty_loss

        # Track the loss
        self._track("kl_penalty_loss", kl_penalty_loss)
        self._track("kl_penalty_scaled", scaled_loss)

        return scaled_loss, shared_loss_data, False

    def _compute_kl_penalty(self, new_logprob: Tensor, old_logprob: Tensor) -> Tensor:
        """Compute KL divergence penalty between old and new policy."""

        # Compute log ratio with clamping to prevent numerical issues
        logratio = torch.clamp(
            new_logprob - old_logprob,
            -self.loss_cfg.max_log_ratio,
            self.loss_cfg.max_log_ratio
        )

        # Importance sampling ratio
        ratio = logratio.exp()

        # Approximate KL divergence: KL ≈ (ratio - 1) - log(ratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        return approx_kl

    def _track(self, key: str, value: Tensor) -> None:
        """Track loss values for monitoring."""
        self.loss_tracker[key].append(float(value.item()))