"""Loss computation functions for PPO training."""

import logging
from typing import Any

import torch
from torch import Tensor

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_state import PolicyState
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


class Losses:
    def __init__(self):
        self.zero()

    def zero(self):
        """Reset all loss values to 0.0."""
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.approx_kl_sum = 0.0
        self.clipfrac_sum = 0.0
        self.l2_reg_loss_sum = 0.0
        self.l2_init_loss_sum = 0.0
        self.ks_action_loss_sum = 0.0
        self.ks_value_loss_sum = 0.0
        self.importance_sum = 0.0
        self.current_logprobs_sum = 0.0
        self.explained_variance = 0.0
        self.minibatches_processed = 0

    def stats(self) -> dict[str, float]:
        """Convert losses to dictionary with proper averages."""
        n = max(1, self.minibatches_processed)

        return {
            "policy_loss": self.policy_loss_sum / n,
            "value_loss": self.value_loss_sum / n,
            "entropy": self.entropy_sum / n,
            "approx_kl": self.approx_kl_sum / n,
            "clipfrac": self.clipfrac_sum / n,
            "l2_reg_loss": self.l2_reg_loss_sum / n,
            "l2_init_loss": self.l2_init_loss_sum / n,
            "ks_action_loss": self.ks_action_loss_sum / n,
            "ks_value_loss": self.ks_value_loss_sum / n,
            "importance": self.importance_sum / n,
            "explained_variance": self.explained_variance,
            "current_logprobs": self.current_logprobs_sum / n,
        }


def process_minibatch_update(
    policy: PolicyAgent,
    experience: Experience,
    minibatch: dict[str, Tensor],
    trainer_cfg: Any,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    obs = minibatch["obs"]

    lstm_state = PolicyState()
    _, new_logprobs, entropy, newvalue, full_logprobs = policy(obs, lstm_state, action=minibatch["actions"])

    new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
    logratio = new_logprobs - minibatch["logprobs"]
    importance_sampling_ratio = logratio.exp()
    experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

    # Re-compute advantages with new ratios (V-trace)
    adv = compute_advantage(
        minibatch["values"],
        minibatch["rewards"],
        minibatch["dones"],
        importance_sampling_ratio,
        minibatch["advantages"],
        trainer_cfg.ppo.gamma,
        trainer_cfg.ppo.gae_lambda,
        trainer_cfg.vtrace.vtrace_rho_clip,
        trainer_cfg.vtrace.vtrace_c_clip,
        device,
    )

    # Normalize advantages with distributed support, then apply prioritized weights
    adv = normalize_advantage_distributed(adv, trainer_cfg.ppo.norm_adv)
    adv = minibatch["prio_weights"] * adv

    # Compute losses
    from metta.rl.ppo import compute_ppo_losses

    pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
        minibatch, new_logprobs, entropy, newvalue, importance_sampling_ratio, adv, trainer_cfg, device
    )

    # Kickstarter losses
    ks_action_loss, ks_value_loss = kickstarter.loss(agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[])

    # L2 init loss
    l2_init_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if trainer_cfg.ppo.l2_init_loss_coef > 0:
        l2_init_loss = trainer_cfg.ppo.l2_init_loss_coef * policy.l2_init_loss().to(device)

    # Total loss
    loss = (
        pg_loss
        - trainer_cfg.ppo.ent_coef * entropy_loss
        + v_loss * trainer_cfg.ppo.vf_coef
        + l2_init_loss
        + ks_action_loss
        + ks_value_loss
    )

    # Update values in experience buffer
    experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

    # Update loss tracking
    losses.policy_loss_sum += pg_loss.item()
    losses.value_loss_sum += v_loss.item()
    losses.entropy_sum += entropy_loss.item()
    losses.approx_kl_sum += approx_kl.item()
    losses.clipfrac_sum += clipfrac.item()
    losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
    losses.ks_action_loss_sum += ks_action_loss.item()
    losses.ks_value_loss_sum += ks_value_loss.item()
    losses.importance_sum += importance_sampling_ratio.mean().item()
    losses.minibatches_processed += 1
    losses.current_logprobs_sum += new_logprobs.mean().item()

    return loss
