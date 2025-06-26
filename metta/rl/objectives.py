"""RL loss objectives, inspired by torch.rl"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience
from metta.rl.functional_trainer import (
    compute_advantage,
    normalize_advantage_distributed,
)
from metta.rl.losses import Losses


class ClipPPOLoss(nn.Module):
    """Clipped PPO loss function.

    This module implements the PPO loss function with clipping, as described in
    the PPO paper. It handles policy loss, value loss, and entropy bonus.
    """

    def __init__(
        self,
        policy: nn.Module,
        vf_coef: float,
        ent_coef: float,
        clip_coef: float,
        vf_clip_coef: float,
        norm_adv: bool,
        clip_vloss: bool,
        gamma: float,
        gae_lambda: float,
        vtrace_rho_clip: float,
        vtrace_c_clip: float,
        l2_reg_loss_coef: float = 0.0,
        l2_init_loss_coef: float = 0.0,
        kickstarter: Optional[Any] = None,
    ):
        super().__init__()
        self.policy = policy
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.vf_clip_coef = vf_clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vtrace_rho_clip = vtrace_rho_clip
        self.vtrace_c_clip = vtrace_c_clip
        self.l2_reg_loss_coef = l2_reg_loss_coef
        self.l2_init_loss_coef = l2_init_loss_coef
        self.kickstarter = kickstarter

    def forward(
        self,
        minibatch: Dict[str, torch.Tensor],
        experience: Experience,
        losses: Losses,
        agent_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Computes the PPO loss for a given minibatch."""
        obs = minibatch["obs"]
        lstm_state = PolicyState()
        _, new_logprobs, entropy, newvalue, full_logprobs = self.policy(obs, lstm_state, action=minibatch["actions"])

        new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
        logratio = new_logprobs - minibatch["logprobs"]
        importance_sampling_ratio = logratio.exp()
        experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

        with torch.no_grad():
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.clip_coef).float().mean()

        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            self.gamma,
            self.gae_lambda,
            self.vtrace_rho_clip,
            self.vtrace_c_clip,
            device,
        )

        adv = normalize_advantage_distributed(adv, self.norm_adv)
        adv = minibatch["prio_weights"] * adv

        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(importance_sampling_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
            v_clipped = minibatch["values"] + torch.clamp(
                newvalue_reshaped - minibatch["values"],
                -self.vf_clip_coef,
                self.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

        entropy_loss = entropy.mean()
        ks_action_loss, ks_value_loss = torch.tensor(0.0), torch.tensor(0.0)
        if self.kickstarter is not None:
            ks_action_loss, ks_value_loss = self.kickstarter.loss(agent_step, full_logprobs, newvalue, obs, [])

        l2_reg_loss = torch.tensor(0.0)
        if self.l2_reg_loss_coef > 0 and hasattr(self.policy, "l2_reg_loss"):
            l2_reg_loss = self.l2_reg_loss_coef * self.policy.l2_reg_loss().to(device)

        l2_init_loss = torch.tensor(0.0)
        if self.l2_init_loss_coef > 0 and hasattr(self.policy, "l2_init_loss"):
            l2_init_loss = self.l2_init_loss_coef * self.policy.l2_init_loss().to(device)

        loss = (
            pg_loss
            - self.ent_coef * entropy_loss
            + v_loss * self.vf_coef
            + l2_reg_loss
            + l2_init_loss
            + ks_action_loss
            + ks_value_loss
        )

        experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))
        losses.policy_loss_sum += pg_loss.item()
        losses.value_loss_sum += v_loss.item()
        losses.entropy_sum += entropy_loss.item()
        losses.approx_kl_sum += approx_kl.item()
        losses.clipfrac_sum += clipfrac.item()
        losses.l2_reg_loss_sum += l2_reg_loss.item() if torch.is_tensor(l2_reg_loss) else l2_reg_loss
        losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
        losses.ks_action_loss_sum += ks_action_loss.item()
        losses.ks_value_loss_sum += ks_value_loss.item()
        losses.importance_sum += importance_sampling_ratio.mean().item()

        return loss
