from typing import Any

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment
from mettagrid.base_config import Config


class GRPOConfig(Config):
    """Configuration for Group Relative Policy Optimization.

    GRPO eliminates the value network and uses group-based advantage estimation.
    During minibatch sampling, returns are normalized within groups of size group_size,
    implementing GRPO's core principle of relative comparison.
    """

    schedule: None = None  # TODO: Implement this

    # GRPO hyperparameters
    # Clip coefficient for policy gradient
    clip_coef: float = Field(default=0.2, gt=0, le=1.0)
    # Entropy regularization weight (higher for more exploration and stability)
    ent_coef: float = Field(default=0.05, ge=0)
    # Discount factor for returns
    gamma: float = Field(default=0.99, ge=0, le=1.0)
    # Group size for relative normalization (returns normalized within groups)
    group_size: int = Field(default=4, gt=1)

    # Training parameters
    # Gradient clipping
    max_grad_norm: float = Field(default=0.5, gt=0)
    # L2 regularization
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization
    # Advantage normalization toggle
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
        loss_config: Any,
    ):
        """Points to the GRPO class for initialization."""
        return GRPO(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


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
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_config)
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
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is required for GRPO rollout")
        self.replay.store(data_td=td, env_id=env_slice)

        return

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """GRPO training loop with group-based advantage estimation."""
        config = self.loss_cfg
        stop_update_epoch = False
        self.policy.reset_memory()
        self.burn_in_steps_iter = 0

        if config.target_kl is not None and mb_idx > 0:
            avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if avg_kl > config.target_kl:
                stop_update_epoch = True

        # On the first minibatch, compute group-based advantages
        if mb_idx == 0:
            self.advantages = self._compute_group_advantages(context)

        # Sample from the buffer
        minibatch, indices = self._sample_minibatch()

        shared_loss_data["sampled_mb"] = minibatch
        shared_loss_data["indices"] = NonTensorData(indices)

        # Forward the policy using the sampled minibatch
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        flat_actions = minibatch["actions"].reshape(B * TT, -1)

        policy_td = self.policy.forward(policy_td, action=flat_actions)
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)

        # Calculate the loss
        loss = self._process_minibatch_update(
            minibatch=minibatch,
            policy_td=policy_td,
            indices=indices,
        )

        return loss, shared_loss_data, stop_update_epoch

    def on_train_phase_end(self, context: ComponentContext) -> None:
        """Track metrics at the end of training phase."""
        pass

    def _compute_group_advantages(self, context: ComponentContext) -> Tensor:
        """Compute discounted returns for GRPO.

        Unlike traditional GRPO which normalizes raw rewards, we compute discounted
        returns here and perform group-based normalization during minibatch sampling.
        This maintains temporal credit assignment while applying GRPO's group comparison.
        """
        cfg = self.loss_cfg
        with torch.no_grad():
            rewards = self.replay.buffer["rewards"]  # [B, T]
            dones = self.replay.buffer["dones"]

            # Compute discounted returns using efficient backward pass
            B, T = rewards.shape
            returns = torch.zeros_like(rewards)
            next_return = torch.zeros(B, device=rewards.device)

            for t in reversed(range(T)):
                # Bootstrap from next timestep, reset at episode boundaries
                next_return = rewards[:, t] + cfg.gamma * next_return * (1.0 - dones[:, t])
                returns[:, t] = next_return

        # Returns will be normalized in groups during minibatch sampling
        return returns

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
    ) -> Tensor:
        """Process minibatch update using GRPO loss."""
        cfg = self.loss_cfg
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]

        importance_sampling_ratio = self._importance_ratio(new_logprob, old_logprob)

        # Get advantages from minibatch (already group-normalized in _sample_minibatch)
        adv = minibatch["advantages"]

        # Optional: apply additional normalization across entire minibatch
        # Note: advantages are already group-normalized, this provides global normalization
        if cfg.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Compute GRPO loss (policy only, no value loss)
        pg_loss, entropy_loss, approx_kl, clipfrac = self.compute_grpo_loss(
            new_logprob,
            old_logprob,
            entropy,
            importance_sampling_ratio,
            adv,
        )

        loss = pg_loss - cfg.ent_coef * entropy_loss

        # Update loss tracking
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
            1 - self.loss_cfg.clip_coef,
            1 + self.loss_cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - old_logprob
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()

        return pg_loss, entropy_loss, approx_kl, clipfrac

    def _sample_minibatch(self) -> tuple[TensorDict, Tensor]:
        """Sample a minibatch and compute group-normalized advantages.

        This implements GRPO's core innovation: comparing returns within groups.
        We sample uniformly, then normalize returns within groups of size group_size.
        """
        cfg = self.loss_cfg
        num_segments = self.replay.buffer.shape[0]
        idx = torch.randint(0, num_segments, (self.replay.minibatch_segments,), device=self.device)

        minibatch = self.replay.buffer[idx]

        with torch.no_grad():
            # Get returns for sampled indices
            returns = self.advantages[idx]  # [minibatch_segments, bptt_horizon]
            B, T = returns.shape

            # Flatten for group processing
            returns_flat = returns.reshape(-1)
            advantages_flat = torch.zeros_like(returns_flat)

            # Group normalization: divide samples into groups and normalize within each group
            group_size = cfg.group_size
            num_complete_groups = len(returns_flat) // group_size
            remainder = len(returns_flat) % group_size

            # Process complete groups
            for g in range(num_complete_groups):
                start_idx = g * group_size
                end_idx = start_idx + group_size
                group_returns = returns_flat[start_idx:end_idx]

                # Normalize within group: A = (R - mean(R)) / std(R)
                group_mean = group_returns.mean()
                group_std = group_returns.std()
                # Only normalize if std is meaningful (> 1e-6), otherwise just center
                if group_std > 1e-6:
                    advantages_flat[start_idx:end_idx] = (group_returns - group_mean) / group_std
                else:
                    advantages_flat[start_idx:end_idx] = group_returns - group_mean

            # Handle remainder samples (if minibatch size not divisible by group_size)
            if remainder > 0:
                start_idx = num_complete_groups * group_size
                remainder_returns = returns_flat[start_idx:]
                remainder_mean = remainder_returns.mean()
                remainder_std = remainder_returns.std()
                # Only normalize if std is meaningful, otherwise just center
                if remainder > 1 and remainder_std > 1e-6:
                    advantages_flat[start_idx:] = (remainder_returns - remainder_mean) / remainder_std
                else:
                    # Single sample or zero variance: just center it
                    advantages_flat[start_idx:] = remainder_returns - remainder_mean

            # Reshape back to [B, T]
            minibatch["advantages"] = advantages_flat.reshape(B, T)

        return minibatch.clone(), idx

    def _importance_ratio(self, new_logprob: Tensor, old_logprob: Tensor) -> Tensor:
        logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
        return logratio.exp()

    def _track(self, key: str, value: Tensor) -> None:
        self.loss_tracker[key].append(float(value.item()))
