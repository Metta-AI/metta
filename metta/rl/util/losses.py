"""Loss computation functions for PPO training."""

import logging
from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed

logger = logging.getLogger(__name__)


class Loss:
    def __init__(self, policy: torch.nn.Module, trainer_cfg: Any, device: torch.device, losses: Losses):
        self.policy = policy
        self.trainer_cfg = trainer_cfg
        self.device = device
        self.losses = losses

    def get_experience_spec(self) -> TensorDict:
        raise NotImplementedError("get_experience_spec not implemented for base class")


class PPO(Loss):
    def __init__(
        self,
        policy: torch.nn.Module,
        experience: Optional[Experience],
        trainer_cfg: Any,
        device: torch.device,
        losses: Losses,
    ):
        super().__init__(policy, trainer_cfg, device, losses)
        self.experience = experience

    def get_experience_spec(self) -> TensorDict:
        return TensorDict(
            {
                "obs": torch.zeros(*self.policy.agent_attributes["obs_shape"], dtype=torch.uint8),
                "rewards": torch.zeros((), dtype=torch.float32),
                "dones": torch.zeros((), dtype=torch.float32),
                "truncateds": torch.zeros((), dtype=torch.float32),
                "actions": torch.zeros(self.policy.agent_attributes["action_space"].shape, dtype=torch.int32),
                "logprobs": torch.zeros((), dtype=torch.float32),
                "values": torch.zeros((), dtype=torch.float32),
            },
            batch_size=[],
        )

    def __call__(
        self,
        td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
        kickstarter: Any,
        agent_step: int,
    ) -> Tensor:
        """Process a single minibatch update and return the total loss."""
        # The policy's training forward pass returns a TD with required tensors for loss calculation.
        new_td = self.policy(td, action=td["actions"])
        new_logprobs = new_td["action_log_prob"].reshape(td["logprobs"].shape)
        entropy = new_td["entropy"]
        newvalue = new_td["value"]
        full_logprobs = new_td["log_probs"]

        logratio = new_logprobs - td["logprobs"]
        importance_sampling_ratio = logratio.exp()

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            td["values"],
            td["rewards"],
            td["dones"],
            importance_sampling_ratio,
            td["advantages"],
            self.trainer_cfg.ppo.gamma,
            self.trainer_cfg.ppo.gae_lambda,
            self.trainer_cfg.vtrace.vtrace_rho_clip,
            self.trainer_cfg.vtrace.vtrace_c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.trainer_cfg.ppo.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
            td,
            new_logprobs,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
            self.trainer_cfg,
        )

        # Kickstarter losses
        ks_action_loss, ks_value_loss = kickstarter.loss(
            agent_step,
            full_logprobs,
            newvalue,
            td["env_obs"],
            teacher_lstm_state=[],
        )

        # L2 init loss
        l2_init_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.trainer_cfg.ppo.l2_init_loss_coef > 0:
            l2_init_loss = self.trainer_cfg.ppo.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)

        # Total loss
        loss = (
            pg_loss
            - self.trainer_cfg.ppo.ent_coef * entropy_loss
            + v_loss * self.trainer_cfg.ppo.vf_coef
            + l2_init_loss
            + ks_action_loss
            + ks_value_loss
        )

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(td["values"].shape), "ratio": importance_sampling_ratio},
            batch_size=td.batch_size,
        )
        self.experience.update(indices, update_td)

        # Update loss tracking
        self.losses.policy_loss_sum += pg_loss.item()
        self.losses.value_loss_sum += v_loss.item()
        self.losses.entropy_sum += entropy_loss.item()
        self.losses.approx_kl_sum += approx_kl.item()
        self.losses.clipfrac_sum += clipfrac.item()
        self.losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
        self.losses.ks_action_loss_sum += ks_action_loss.item()
        self.losses.ks_value_loss_sum += ks_value_loss.item()
        self.losses.importance_sum += importance_sampling_ratio.mean().item()
        self.losses.minibatches_processed += 1
        self.losses.current_logprobs_sum += new_logprobs.mean().item()

        return loss

    def compute_ppo_losses(
        self,
        minibatch: TensorDict,
        new_logprobs: Tensor,
        entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
        trainer_cfg: Any,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute PPO losses for policy and value functions."""
        # Policy loss
        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio, 1 - trainer_cfg.ppo.clip_coef, 1 + trainer_cfg.ppo.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if trainer_cfg.ppo.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
            v_clipped = minibatch["values"] + torch.clamp(
                newvalue_reshaped - minibatch["values"],
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprobs - minibatch["logprobs"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


class Contrastive(Loss):
    def __init__(
        self,
        policy: torch.nn.Module,
        experience: Optional[Experience],
        trainer_cfg: Any,
        device: torch.device,
        losses: Losses,
    ):
        super().__init__(policy, trainer_cfg, device, losses)
        self.experience = experience

    def get_experience_spec(self) -> TensorDict:
        pass

    # def __call__(
    #     self, td: TensorDict, indices: Tensor, prio_weights: Tensor, kickstarter: Any, agent_step: int
    # ) -> Tensor:
    #     pos_example = self.experience.buffer["obs"][indices + 1]
    #     neg_example = self.experience.buffer["obs"][indices - 1]
    #     td = self.experience.buffer[indices]["obs"]

    #     new_td = self.policy.components["pred_output"](td)
    #     pred = new_td["pred"]

    # loss = cross entropy between pred and pos_example

    # compute loss between new_td keys
