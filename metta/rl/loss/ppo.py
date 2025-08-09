from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.base_loss import BaseLoss, LossTracker
from metta.rl.ppo import compute_ppo_losses
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState
from metta.utils.batch import calculate_prioritized_sampling_params


class PPO(BaseLoss):
    def __init__(
        self,
        policy: PolicyAgent,
        cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
    ):
        super().__init__(policy, cfg, vec_env, device, loss_tracker)
        # Placeholder; real advantages tensor is computed from replay on first minibatch
        self.advantages = torch.tensor(0.0, device=self.device)
        self.anneal_beta = 0.0

    def get_experience_spec(self, nvec: list[int] | torch.Tensor, act_dtype: torch.dtype) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=(), dtype=torch.float32)

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
            returns=scalar_f32,
        )

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        # # early exit if kl divergence is too high
        # if trainer_state[3] == self.policy.replay.num_minibatches - 1:  # av check this
        #     # Early exit if KL divergence is too high
        #     if self.cfg.ppo.target_kl is not None:
        #         average_approx_kl = self.loss_tracker.approx_kl_sum / self.loss_tracker.minibatches_processed
        #         if average_approx_kl > self.cfg.ppo.target_kl:
        #             self.early_exit = True
        # Early exit if KL divergence is too high
        if self.cfg.ppo.target_kl is not None and self.loss_tracker.minibatches_processed > 0:
            average_approx_kl = self.loss_tracker.approx_kl_sum / self.loss_tracker.minibatches_processed
            if average_approx_kl > self.cfg.ppo.target_kl:
                trainer_state.early_stop_update_epoch = True

        # On first minibatch of the update epoch, compute advantages and sampling params
        if trainer_state.mb_idx == 0:
            self.advantages, self.anneal_beta = self.first_mb(trainer_state)

        # Sample minibatch
        minibatch, indices, prio_weights = self.policy.replay.sample_minibatch(
            advantages=self.advantages,
            prio_alpha=self.cfg.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
        )

        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))

        loss = self.process_minibatch_update(
            policy=self.policy,
            minibatch=minibatch,
            policy_td=policy_td,
            indices=indices,
            prio_weights=prio_weights,
        )
        # # on last mb, do this
        # if (
        #     trainer_state[2] == self.cfg.update_epochs
        # ):  # av check this increments properly. might need "on update epoch"
        # Calculate explained variance
        # On last minibatch of the update epoch, compute explained variance
        if trainer_state.is_last_minibatch():
            y_pred = self.policy.replay.buffer["values"].flatten()
            y_true = self.advantages.flatten() + self.policy.replay.buffer["values"].flatten()
            var_y = y_true.var()
            self.loss_tracker.explained_variance = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0

        # av write to shared_loss_data

        return loss, shared_loss_data

    def first_mb(self, trainer_state: TrainerState) -> tuple[Tensor, float]:
        anneal_beta = calculate_prioritized_sampling_params(
            epoch=trainer_state.epoch,
            total_timesteps=self.cfg.total_timesteps,
            batch_size=self.cfg.batch_size,
            prio_alpha=self.cfg.prioritized_experience_replay.prio_alpha,
            prio_beta0=self.cfg.prioritized_experience_replay.prio_beta0,
        )

        # Compute initial advantages
        advantages = torch.zeros(self.policy.replay.buffer["values"].shape, device=self.device)
        initial_importance_sampling_ratio = torch.ones_like(self.policy.replay.buffer["values"])

        advantages = compute_advantage(
            self.policy.replay.buffer["values"],
            self.policy.replay.buffer["rewards"],
            self.policy.replay.buffer["dones"],
            initial_importance_sampling_ratio,
            advantages,
            self.cfg.ppo.gamma,
            self.cfg.ppo.gae_lambda,
            self.cfg.vtrace.vtrace_rho_clip,
            self.cfg.vtrace.vtrace_c_clip,
            self.device,
        )

        return advantages, anneal_beta

    def process_minibatch_update(
        self,
        policy: PolicyAgent,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
    ) -> Tensor:
        """Process a single minibatch update and return the total loss."""
        policy_td = policy(policy_td, action=minibatch["actions"])

        old_act_log_prob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_act_log_prob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["value"]

        logratio = new_logprob - old_act_log_prob
        importance_sampling_ratio = logratio.exp()

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            self.cfg.ppo.gamma,
            self.cfg.ppo.gae_lambda,
            self.cfg.vtrace.vtrace_rho_clip,
            self.cfg.vtrace.vtrace_c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.cfg.ppo.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
            minibatch,
            new_logprob,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
            self.cfg,
        )

        loss = pg_loss - self.cfg.ppo.ent_coef * entropy_loss + v_loss * self.cfg.ppo.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.policy.replay.update(indices, update_td)

        # Update loss tracking
        self.loss_tracker.policy_loss_sum += pg_loss.item()
        self.loss_tracker.value_loss_sum += v_loss.item()
        self.loss_tracker.entropy_sum += entropy_loss.item()
        self.loss_tracker.approx_kl_sum += approx_kl.item()
        self.loss_tracker.clipfrac_sum += clipfrac.item()
        self.loss_tracker.importance_sum += importance_sampling_ratio.mean().item()
        self.loss_tracker.minibatches_processed += 1
        self.loss_tracker.current_logprobs_sum += new_logprob.mean().item()

        return loss
