from typing import Any, Tuple

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.loss import Loss
from metta.rl.training.component_context import ComponentContext
from metta.rl.training.training_environment import TrainingEnvironment
from metta.utils.batch import calculate_prioritized_sampling_params
from mettagrid.config import Config


class PrioritizedExperienceReplayConfig(Config):
    # Alpha=0 means uniform sampling; tuned via sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta baseline per Schaul et al. (2016)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(Config):
    # Defaults follow IMPALA (Espeholt et al., 2018)
    rho_clip: float = Field(default=1.0, gt=0)
    c_clip: float = Field(default=1.0, gt=0)


class KLPenaltyConfig(Config):
    """Configuration for PPO with KL penalty instead of clipping."""

    # KL penalty coefficient (replaces clip_coef)
    kl_penalty_coef: float = Field(default=0.01, ge=0)
    # Entropy term weight from sweep
    ent_coef: float = Field(default=0.010000, ge=0)
    # GAE lambda tuned via sweep (cf. standard 0.95)
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)
    # Gamma tuned for shorter effective horizon
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping default
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value clipping mirrors policy clip
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value term weight from sweep
    vf_coef: float = Field(default=0.897619, ge=0)
    # L2 regularization defaults to disabled
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    # Advantage normalization toggle
    norm_adv: bool = True
    # Value loss clipping toggle
    clip_vloss: bool = True
    # Target KL for early stopping (None disables)
    target_kl: float | None = None
    # Clamp log ratio to prevent numerical issues
    max_log_ratio: float = Field(default=10.0, gt=0)

    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

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
            instance_name,
            loss_cfg=self,
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
        logratio = torch.clamp(new_logprob - old_logprob, -self.loss_cfg.max_log_ratio, self.loss_cfg.max_log_ratio)

        # Importance sampling ratio
        ratio = logratio.exp()

        # Approximate KL divergence: KL ≈ (ratio - 1) - log(ratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        return approx_kl

    def _track(self, key: str, value: Tensor) -> None:
        """Track loss values for monitoring."""
        if self.loss_tracker is not None:
            self.loss_tracker[key].append(float(value.item()))
