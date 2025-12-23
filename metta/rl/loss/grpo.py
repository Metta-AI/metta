from typing import Any, Optional

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class GRPOConfig(LossConfig):
    """Configuration for Group Relative Policy Optimization."""

    # Clip coefficient for policy gradient
    clip_coef: float = Field(default=0.2, gt=0, le=1.0)
    # Entropy regularization weight
    ent_coef: float = Field(default=0.01, ge=0)
    # Discount factor for returns
    gamma: float = Field(default=0.99, ge=0, le=1.0)
    # Number of responses to sample per prompt for group comparison
    group_size: int = Field(default=4, gt=1)

    # Training parameters
    # Gradient clipping
    max_grad_norm: float = Field(default=0.5, gt=0)
    # L2 regularization
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Advantage normalization
    norm_adv: bool = True
    # Target KL for early stopping (None disables)
    target_kl: float | None = None

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "GRPO":
        return GRPO(policy, trainer_cfg, env, device, instance_name, self)


class GRPO(Loss):
    """Group Relative Policy Optimization loss.

    GRPO eliminates the value network and uses group-based advantage estimation,
    where advantages are computed relative to the mean reward of a group of
    sampled responses for each state/prompt.
    """

    __slots__ = (
        "advantages",
        "burn_in_steps",
        "burn_in_steps_iter",
        "last_action",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: "GRPOConfig",
    ) -> None:
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.burn_in_steps = 0
        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        self.burn_in_steps_iter = 0
        self.last_action = None
        self.register_state_attr("burn_in_steps_iter")

    def get_experience_spec(self) -> Composite:
        """Get experience specification without value predictions."""
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            act_log_prob=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Run policy rollout without value prediction."""
        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # Store experience
        env_slice = self._training_env_id(
            context, error="ComponentContext.training_env_id is required for GRPO rollout"
        )
        self.replay.store(data_td=td, env_id=env_slice)

        return

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"act_log_prob", "entropy"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """GRPO training loop with group-based advantage estimation."""
        config = self.cfg
        stop_update_epoch = False
        self.policy.reset_memory()
        self.burn_in_steps_iter = 0

        if config.target_kl is not None and mb_idx > 0:
            avg_kl = self.metric_mean("approx_kl")
            if avg_kl > config.target_kl:
                stop_update_epoch = True

        if mb_idx == 0:
            # overwrite advantages in the replay buffer with the group-based advantages
            self.replay.buffer["advantages"] = self._compute_group_advantages(context)
            self.replay.buffer["advantages_full"] = self.replay.buffer["advantages"]  # maybe we don't need this?
            indices = shared_loss_data["indices"][:, 0]
            shared_loss_data["advantages"] = self.replay.buffer["advantages"][indices]

        loss = self._process_minibatch_update(
            minibatch=shared_loss_data["sampled_mb"],
            policy_td=shared_loss_data["policy_td"],
            indices=shared_loss_data["indices"][:, 0],
        )

        return loss, shared_loss_data, stop_update_epoch

    def on_train_phase_end(self, context: ComponentContext) -> None:
        """Track metrics at the end of training phase."""
        pass

    def _compute_group_advantages(self, context: ComponentContext) -> Tensor:
        """Compute group-based advantages relative to mean reward.

        In GRPO, we compute advantages by comparing each trajectory's
        discounted return against the mean return of a group of trajectories.
        This eliminates the need for a value network.
        """
        cfg = self.cfg
        with torch.no_grad():
            # Compute discounted returns for all trajectories
            returns = self._compute_returns(
                self.replay.buffer["rewards"],
                self.replay.buffer["dones"],
                cfg.gamma,
            )

            # Group trajectories and compute advantages relative to group mean
            B = returns.shape[0]
            group_size = min(cfg.group_size, B)

            # Reshape to [num_groups, group_size, seq_len]
            num_groups = B // group_size
            if num_groups == 0:
                # Fallback: if we don't have enough samples, use global mean
                mean_return = returns.mean(dim=0, keepdim=True)
                advantages = returns - mean_return
            else:
                # Trim to divisible by group_size
                returns_grouped = returns[: num_groups * group_size].reshape(num_groups, group_size, -1)

                # Compute group mean for each group
                group_means = returns_grouped.mean(dim=1, keepdim=True)  # [num_groups, 1, seq_len]

                # Advantages are relative to group mean
                advantages_grouped = returns_grouped - group_means

                # Flatten back to [B', seq_len]
                advantages = advantages_grouped.reshape(num_groups * group_size, -1)

                # Handle remaining samples with global mean
                if B > num_groups * group_size:
                    remaining = returns[num_groups * group_size :]
                    remaining_mean = remaining.mean(dim=0, keepdim=True)
                    remaining_adv = remaining - remaining_mean
                    advantages = torch.cat([advantages, remaining_adv], dim=0)

        return advantages

    def _compute_returns(self, rewards: Tensor, dones: Tensor, gamma: float) -> Tensor:
        """Compute discounted returns for each trajectory."""
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=rewards.device)

        # Compute returns backward through time
        for t in reversed(range(T)):
            running_return = rewards[:, t] + gamma * running_return * (1 - dones[:, t])
            returns[:, t] = running_return

        return returns

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
    ) -> Tensor:
        """Process minibatch update using GRPO loss."""
        cfg = self.cfg
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]

        importance_sampling_ratio = self._importance_ratio(new_logprob, old_logprob)

        adv = minibatch["advantages"]

        # Normalize advantages
        if cfg.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss, entropy_loss, approx_kl, clipfrac = self.compute_grpo_loss(
            new_logprob,
            old_logprob,
            entropy,
            importance_sampling_ratio,
            adv,
        )

        loss = pg_loss - cfg.ent_coef * entropy_loss

        self._track("policy_loss", pg_loss)
        self._track("entropy", entropy_loss)
        self._track("approx_kl", approx_kl)
        self._track("clipfrac", clipfrac)
        self._track("importance", importance_sampling_ratio.mean())
        self._track("current_logprobs", new_logprob.mean())

        return loss

    def compute_grpo_loss(
        self,
        new_logprob: Tensor,
        old_logprob: Tensor,
        entropy: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute GRPO loss (policy gradient only, no value loss)."""
        # Clipped policy gradient loss
        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio,
            1 - self.cfg.clip_coef,
            1 + self.cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - old_logprob
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()

        return pg_loss, entropy_loss, approx_kl, clipfrac

    def _importance_ratio(self, new_logprob: Tensor, old_logprob: Tensor) -> Tensor:
        logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
        return logratio.exp()

    def _track(self, key: str, value: Tensor) -> None:
        self.track_metric(key, value)
