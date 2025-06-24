"""PPO loss computation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from metta.rl.experience import Experience
from metta.rl.losses import Losses


@dataclass
class PPOLossConfig:
    """Configuration for PPO loss computation."""

    clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    vf_clip_coef: float = 0.1
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0


def compute_ppo_loss(
    agent,
    experience: Experience,
    config: PPOLossConfig,
    kickstarter=None,
) -> Tuple[torch.Tensor, Losses]:
    """Compute PPO loss for a batch of experience.

    Args:
        agent: The policy/agent to compute losses for
        experience: Experience buffer with trajectories
        config: PPO loss configuration
        kickstarter: Optional kickstarter for distillation

    Returns:
        Tuple of (total_loss, loss_components)
    """
    losses = Losses()

    # Compute advantages
    advantages = compute_advantages(
        experience.values,
        experience.rewards,
        experience.dones,
        experience.ratio,
        config.gamma,
        config.gae_lambda,
        config.vtrace_rho_clip,
        config.vtrace_c_clip,
    )

    # Normalize advantages if requested
    if config.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Sample minibatch
    mb = experience.sample_minibatch(
        advantages=advantages,
        prio_alpha=0.0,  # TODO: Make configurable
        prio_beta=0.6,  # TODO: Make configurable
        minibatch_idx=0,
        total_minibatches=1,
    )

    # Forward pass through agent
    from metta.agent.policy_state import PolicyState

    state = PolicyState()

    _, newlogprob, entropy, newvalue, aux_losses = agent(mb["obs"], state, action=mb["actions"])

    # Compute ratio for importance sampling
    logratio = newlogprob - mb["logprobs"]
    ratio = logratio.exp()

    # Policy loss (PPO clip)
    pg_loss1 = -mb["advantages"] * ratio
    pg_loss2 = -mb["advantages"] * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    if config.clip_vloss:
        v_loss_unclipped = (newvalue - mb["returns"]) ** 2
        v_clipped = mb["values"] + torch.clamp(
            newvalue - mb["values"],
            -config.vf_clip_coef,
            config.vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - mb["returns"]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue - mb["returns"]) ** 2).mean()

    # Entropy loss
    ent_loss = entropy.mean()

    # Total loss
    loss = pg_loss - config.ent_coef * ent_loss + v_loss * config.vf_coef

    # Add auxiliary losses if any
    total_loss = loss
    if aux_losses is not None and isinstance(aux_losses, dict) and len(aux_losses) > 0:
        for aux_loss in aux_losses.values():
            total_loss = total_loss + aux_loss

    # Add kickstarter losses if applicable
    if kickstarter:
        ks_losses = kickstarter.compute_loss(mb["obs"], mb["actions"], newvalue)
        if ks_losses["action_loss"] is not None:
            loss += ks_losses["action_loss"]
            losses.ks_action_loss = ks_losses["action_loss"].item()
        if ks_losses["value_loss"] is not None:
            loss += ks_losses["value_loss"]
            losses.ks_value_loss = ks_losses["value_loss"].item()

    # Update loss tracking
    losses.policy_loss = pg_loss.item()
    losses.value_loss = v_loss.item()
    losses.entropy = ent_loss.item()

    # Compute KL divergence and clip fraction for monitoring
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > config.clip_coef).float().mean()]
        losses.approx_kl = approx_kl.item()
        losses.clipfrac = clipfracs[0].item()

        # Explained variance
        y_pred, y_true = mb["values"], mb["returns"]
        var_y = torch.var(y_true)
        explained_var = torch.nan_to_num(1 - torch.var(y_true - y_pred) / var_y)
        losses.explained_variance = explained_var.item()

    losses.minibatches_processed = 1

    return total_loss, losses


def compute_advantages(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    importance_sampling_ratio: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> torch.Tensor:
    """Compute advantages using GAE with optional V-trace corrections.

    This is a simplified version - in practice you'd use the optimized C++ version.
    """
    advantages = torch.zeros_like(values)

    # Simple GAE computation (without V-trace for now)
    lastgaelam = 0
    for t in reversed(range(len(rewards[0]))):
        if t == len(rewards[0]) - 1:
            nextnonterminal = 1.0 - dones[:, t]
            nextvalues = values[:, t]
        else:
            nextnonterminal = 1.0 - dones[:, t]
            nextvalues = values[:, t + 1]

        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    return advantages
