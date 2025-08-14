from typing import Any, Tuple

import numpy as np
import torch
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.loss.loss_tracker import LossTracker
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState
from metta.utils.batch import calculate_prioritized_sampling_params


class PPO(BaseLoss):
    __slots__ = (
        "advantages",
        "anneal_beta",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
        policy_store: PolicyStore,
        loss_instance_name: str,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, loss_tracker, policy_store, loss_instance_name)
        self.advantages = torch.tensor(0.0, device=self.device)
        self.anneal_beta = 0.0

    def get_experience_spec(self) -> Composite:
        act_space = self.vec_env.single_action_space
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

    def losses_to_track(self) -> list[str]:
        return [
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "importance",
            "current_logprobs",
            "explained_variance",
        ]

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        with torch.no_grad():
            self.policy(td)

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        self.policy.on_train_mb_start()

        # # early exit if kl divergence is too high
        # if trainer_state[3] == self.policy.replay.num_minibatches - 1:  # av check this
        #     # Early exit if KL divergence is too high
        #     if self.cfg.ppo.target_kl is not None:
        #         average_approx_kl = self.loss_tracker.approx_kl_sum / self.loss_tracker.minibatches_processed
        #         if average_approx_kl > self.cfg.ppo.target_kl:
        #             self.early_exit = True
        # Early exit if KL divergence is too high

        if self.loss_cfg.target_kl is not None and self.loss_tracker.minibatches_processed > 0:
            average_approx_kl = self.loss_tracker.get("approx_kl") / self.loss_tracker.minibatches_processed
            if average_approx_kl > self.loss_cfg.target_kl:
                trainer_state.early_stop_update_epoch = True

        # On first minibatch of the update epoch, compute advantages and sampling params
        if trainer_state.mb_idx == 0:
            self.advantages, self.anneal_beta = self.on_first_mb(trainer_state)

        # Sample minibatch
        minibatch, indices, prio_weights = self.sample_minibatch(
            advantages=self.advantages,
            prio_alpha=self.loss_cfg.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
        )

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(indices)  # av this breaks compile

        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        policy_td = self.policy(policy_td, action=minibatch["actions"])
        shared_loss_data["policy_td"] = policy_td  # write the policy output td for others to reuse

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
            with torch.no_grad():
                y_pred = self.policy.replay.buffer["values"].flatten()
                y_true = self.advantages.flatten() + self.policy.replay.buffer["values"].flatten()
                var_y = y_true.var()
                ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
                self.loss_tracker.set("explained_variance", float(ev))

        # av write to shared_loss_data

        return loss, shared_loss_data

    def on_first_mb(self, trainer_state: TrainerState) -> tuple[Tensor, float]:
        # reset importance sampling ratio
        if "ratio" in self.policy.replay.buffer.keys():
            self.policy.replay.buffer["ratio"].fill_(1.0)

        with torch.no_grad():
            anneal_beta = calculate_prioritized_sampling_params(
                epoch=trainer_state.epoch,
                total_timesteps=self.trainer_cfg.total_timesteps,
                batch_size=self.trainer_cfg.batch_size,
                prio_alpha=self.loss_cfg.prioritized_experience_replay.prio_alpha,
                prio_beta0=self.loss_cfg.prioritized_experience_replay.prio_beta0,
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
                self.loss_cfg.gamma,
                self.loss_cfg.gae_lambda,
                self.loss_cfg.vtrace.rho_clip,
                self.loss_cfg.vtrace.c_clip,
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
            self.loss_cfg.gamma,
            self.loss_cfg.gae_lambda,
            self.loss_cfg.vtrace.rho_clip,
            self.loss_cfg.vtrace.c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.loss_cfg.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
            minibatch,
            new_logprob,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
        )

        loss = pg_loss - self.loss_cfg.ent_coef * entropy_loss + v_loss * self.loss_cfg.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.policy.replay.update(indices, update_td)

        # Update loss tracking
        self.loss_tracker.add("policy_loss", float(pg_loss.item()))
        self.loss_tracker.add("value_loss", float(v_loss.item()))
        self.loss_tracker.add("entropy", float(entropy_loss.item()))
        self.loss_tracker.add("approx_kl", float(approx_kl.item()))
        self.loss_tracker.add("clipfrac", float(clipfrac.item()))
        self.loss_tracker.add("importance", float(importance_sampling_ratio.mean().item()))
        self.loss_tracker.add("current_logprobs", float(new_logprob.mean().item()))

        return loss

    def compute_ppo_losses(
        self,
        minibatch: TensorDict,
        new_logprob: Tensor,
        entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute PPO losses for policy and value functions."""
        # Policy loss
        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio,
            1 - self.loss_cfg.clip_coef,
            1 + self.loss_cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        returns = minibatch["returns"]
        old_values = minibatch["values"]

        # Value loss
        newvalue_reshaped = newvalue.view(returns.shape)
        if self.loss_cfg.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - returns) ** 2
            vf_clip_coef = self.loss_cfg.vf_clip_coef
            v_clipped = old_values.detach() + torch.clamp(
                newvalue_reshaped - old_values.detach(),
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - minibatch["act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

    def sample_minibatch(
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
        idx = torch.multinomial(prio_probs, self.policy.replay.minibatch_segments)

        minibatch = self.policy.replay.buffer[idx]

        with torch.no_grad():
            minibatch["advantages"] = advantages[idx]
            minibatch["returns"] = advantages[idx] + minibatch["values"]
            prio_weights = (self.policy.replay.segments * prio_probs[idx, None]) ** -prio_beta
        return minibatch.clone(), idx, prio_weights
