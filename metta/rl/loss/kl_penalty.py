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
    d_target_kl: float = Field(default=0.01, ge=0)
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
    """PPO with KL penalty instead of clipping."""

    __slots__ = (
        "advantages",
        "anneal_beta",
        "burn_in_steps",
        "burn_in_steps_iter",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_cfg: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_cfg)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.anneal_beta = 0.0
        self.burn_in_steps = 0
        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        self.burn_in_steps_iter = 0
        self.register_state_attr("anneal_beta", "burn_in_steps_iter")

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        nvec = act_space.nvec
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=MultiCategorical(
                nvec=nvec,
                dtype=act_dtype,
            ),
            act_log_prob=scalar_f32,
            values=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is required for PPO rollout")
        self.replay.store(data_td=td, env_id=env_slice)

        return

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """This is the PPO KL penalty algorithm training loop."""
        config = self.loss_cfg
        stop_update_epoch = False
        self.policy.reset_memory()
        self.burn_in_steps_iter = 0
        if config.target_kl is not None and mb_idx > 0:
            avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if avg_kl > config.target_kl:
                stop_update_epoch = True

        # On the first minibatch of the update epoch, compute advantages and sampling params
        if mb_idx == 0:
            self.advantages, self.anneal_beta = self._on_first_mb(context)

        # Then sample from the buffer (this happens at every minibatch)
        minibatch, indices, prio_weights = self._sample_minibatch(
            advantages=self.advantages,
            prio_alpha=config.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
        )

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(indices)  # av this breaks compile

        # Then forward the policy using the sampled minibatch
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        flat_actions = minibatch["actions"].reshape(B * TT, -1)

        policy_td = self.policy.forward(policy_td, action=flat_actions)
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)

        # Finally, calculate the loss!
        loss = self._process_minibatch_update(
            minibatch=minibatch,
            policy_td=policy_td,
            indices=indices,
            prio_weights=prio_weights,
        )

        avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
        # Update β based on adaptive KL mechanism
        if avg_kl < self.loss_cfg.d_target_kl / 1.5:
            self.loss_cfg.kl_penalty_coef /= 2
        elif avg_kl > self.loss_cfg.d_target_kl * 1.5:
            self.loss_cfg.kl_penalty_coef *= 2

        return loss, shared_loss_data, stop_update_epoch

    def on_train_phase_end(self, context: ComponentContext) -> None:
        with torch.no_grad():
            y_pred = self.replay.buffer["values"].flatten()
            y_true = self.advantages.flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

    def _on_first_mb(self, context: ComponentContext) -> tuple[Tensor, float]:
        # reset importance sampling ratio
        if "ratio" in self.replay.buffer.keys():
            self.replay.buffer["ratio"].fill_(1.0)

        cfg = self.loss_cfg
        with torch.no_grad():
            anneal_beta = calculate_prioritized_sampling_params(
                epoch=context.epoch,
                total_timesteps=self.trainer_cfg.total_timesteps,
                batch_size=self.trainer_cfg.batch_size,
                prio_alpha=cfg.prioritized_experience_replay.prio_alpha,
                prio_beta0=cfg.prioritized_experience_replay.prio_beta0,
            )

            advantages = torch.zeros_like(self.replay.buffer["values"], device=self.device)
            advantages = compute_advantage(
                self.replay.buffer["values"],
                self.replay.buffer["rewards"],
                self.replay.buffer["dones"],
                torch.ones_like(self.replay.buffer["values"]),
                advantages,
                cfg.gamma,
                cfg.gae_lambda,
                cfg.vtrace.rho_clip,
                cfg.vtrace.c_clip,
                self.device,
            )

        return advantages, anneal_beta

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

        # Re-compute advantages with new ratios (V-trace)
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

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, cfg.norm_adv)
        adv = prio_weights * adv

        # Compute losses using KL penalty instead of clipping
        pg_loss, v_loss, entropy_loss, approx_kl = self.compute_kl_penalty_losses(
            minibatch,
            new_logprob,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
        )

        loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.replay.update(indices, update_td)

        # Update loss tracking
        self._track("policy_loss", pg_loss)
        self._track("value_loss", v_loss)
        self._track("entropy", entropy_loss)
        self._track("approx_kl", approx_kl)
        self._track("importance", importance_sampling_ratio.mean())
        self._track("current_logprobs", new_logprob.mean())

        return loss

    def compute_kl_penalty_losses(
        self,
        minibatch: TensorDict,
        new_logprob: Tensor,
        entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute losses using KL penalty instead of clipping."""
        # Policy loss with KL penalty
        pg_loss = (-adv * importance_sampling_ratio).mean()

        # Add KL penalty term
        logratio = new_logprob - minibatch["act_log_prob"]
        # Approximate KL divergence: KL ≈ (ratio - 1) - log(ratio)
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()

        pg_loss = pg_loss + self.loss_cfg.kl_penalty_coef * approx_kl

        returns = minibatch["returns"]
        old_values = minibatch["values"]

        # Value loss (same as PPO)
        newvalue_reshaped = newvalue.view(returns.shape)
        if self.loss_cfg.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - returns) ** 2
            vf_clip_coef = self.loss_cfg.vf_clip_coef
            v_clipped = old_values + torch.clamp(
                newvalue_reshaped - old_values,
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        return pg_loss, v_loss, entropy_loss, approx_kl

    def _sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,
    ) -> tuple[TensorDict, Tensor, Tensor]:
        """Sample a prioritized minibatch."""
        # Prioritized sampling based on advantage magnitude
        adv_magnitude = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(prio_probs, self.replay.minibatch_segments)

        minibatch = self.replay.buffer[idx]

        with torch.no_grad():
            minibatch["advantages"] = advantages[idx]
            minibatch["returns"] = advantages[idx] + minibatch["values"]
            prio_weights = (self.replay.segments * prio_probs[idx, None]) ** -prio_beta
        return minibatch.clone(), idx, prio_weights

    def _importance_ratio(self, new_logprob: Tensor, old_logprob: Tensor) -> Tensor:
        logratio = torch.clamp(new_logprob - old_logprob, -self.loss_cfg.max_log_ratio, self.loss_cfg.max_log_ratio)
        return logratio.exp()

    def _track(self, key: str, value: Tensor) -> None:
        """Track loss values for monitoring."""
        if self.loss_tracker is not None:
            self.loss_tracker[key].append(float(value.item()))
