"""Loss computation functions for PPO training."""

import logging
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.rl.util.advantage import compute_advantage, normalize_advantage_distributed

logger = logging.getLogger(__name__)


def compute_ppo_losses(
    minibatch: Dict[str, Tensor],
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute PPO losses for policy and value functions."""
    with torch.profiler.record_function("ppo_losses"):
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


def compute_contrastive_loss(
    minibatch: Dict[str, Tensor],
    lstm_hidden: Tensor,  # [minibatch_segments, bptt_horizon, hidden_size]
    all_lstm_hidden: Tensor,  # [segments, bptt_horizon, hidden_size]
    trainer_cfg: Any,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute contrastive loss using InfoNCE with future state prediction."""

    with torch.profiler.record_function("contrastive_loss"):
        # flatten for easier indexing
        batch_size = lstm_hidden.shape[0] * lstm_hidden.shape[1]
        hidden_size = lstm_hidden.shape[2]
        lstm_hidden_flat = lstm_hidden.view(batch_size, hidden_size)
        all_lstm_flat = all_lstm_hidden.view(-1, hidden_size)

        # get current indices in flattened space
        segment_indices = minibatch["indices"]  # [minibatch_segments]
        bptt_horizon = lstm_hidden.shape[1]
        current_indices = (
            segment_indices.unsqueeze(1) * bptt_horizon + torch.arange(bptt_horizon, device=device)
        ).view(-1)

        # sample future indices using geometric distribution
        u = torch.rand(batch_size, device=device)
        gamma = torch.tensor(trainer_cfg.contrastive.gamma, device=device)  # Convert to tensor
        delta = torch.floor(torch.log(1 - u) / torch.log(gamma)).long()
        delta = torch.clamp(delta, min=1, max=bptt_horizon - 1)

        # adjust delta for episode boundaries
        for i in range(batch_size):
            seg_idx = i // bptt_horizon
            time_idx = i % bptt_horizon

            # find next done after current timestep
            future_dones = minibatch["dones"][seg_idx, time_idx:]
            done_indices = torch.where(future_dones)[0]

            if len(done_indices) > 0:
                # can't sample beyond episode boundary
                max_steps_available = done_indices[0].item()
                delta[i] = min(delta[i].item(), max_steps_available)

        positive_indices = current_indices + delta

        # Ensure positive indices don't exceed buffer bounds
        max_valid_index = all_lstm_flat.shape[0] - 1
        positive_indices = torch.clamp(positive_indices, min=0, max=max_valid_index)

        # sample negatives
        num_negatives = trainer_cfg.contrastive.num_negatives
        negative_indices = torch.randint(0, all_lstm_flat.shape[0], (batch_size, num_negatives), device=device)

        # ensure negatives aren't the current or positive indices
        for i in range(batch_size):
            mask = (negative_indices[i] == current_indices[i]) | (negative_indices[i] == positive_indices[i])
            while mask.any():
                num_to_replace = mask.sum().item()
                if num_to_replace > 0:
                    new_indices = torch.randint(0, all_lstm_flat.shape[0], size=(num_to_replace,), device=device)
                    negative_indices[i, mask] = new_indices
                mask = (negative_indices[i] == current_indices[i]) | (negative_indices[i] == positive_indices[i])

        # get states
        current_states = lstm_hidden_flat
        positive_states = all_lstm_flat[positive_indices]
        negative_states = all_lstm_flat[negative_indices]  # [batch_size, num_negatives, hidden_size]

        # normalize for cosine similarity
        current_norm = torch.nn.functional.normalize(current_states, dim=-1)
        positive_norm = torch.nn.functional.normalize(positive_states, dim=-1)
        negative_norm = torch.nn.functional.normalize(negative_states, dim=-1)

        # compute similarities
        temperature = trainer_cfg.contrastive.temperature
        pos_sim = (current_norm * positive_norm).sum(dim=-1) / temperature
        neg_sim = torch.sum(current_norm.unsqueeze(1) * negative_norm, dim=-1) / temperature

        # infonce loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        infonce_loss = torch.nn.functional.cross_entropy(logits, labels)

        # logsumexp regularization
        logsumexp_reg = trainer_cfg.contrastive.logsumexp_coef * torch.logsumexp(neg_sim, dim=-1).mean()

        contrastive_loss = infonce_loss + logsumexp_reg

        # compute metrics
        metrics = {
            "contrastive_loss": contrastive_loss.item(),
            "contrastive_infonce": infonce_loss.item(),
            "contrastive_logsumexp": logsumexp_reg.item(),
            "contrastive_pos_sim": pos_sim.mean().item(),
            "contrastive_neg_sim": neg_sim.mean().item(),
        }

    return contrastive_loss, metrics


def process_minibatch_update(
    policy: torch.nn.Module,
    experience: Experience,
    minibatch: Dict[str, Tensor],
    advantages: Tensor,
    trainer_cfg: Any,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    with torch.profiler.record_function("minibatch_update"):
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
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
            minibatch, new_logprobs, entropy, newvalue, importance_sampling_ratio, adv, trainer_cfg, device
        )

        # Kickstarter losses
        ks_action_loss, ks_value_loss = kickstarter.loss(
            agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[]
        )

        # L2 init loss
        l2_init_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if trainer_cfg.ppo.l2_init_loss_coef > 0:
            l2_init_loss = trainer_cfg.ppo.l2_init_loss_coef * policy.l2_init_loss()

        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_metrics = {}

        if trainer_cfg.contrastive.enabled and hasattr(experience, "lstm_outputs"):
            minibatch_lstm = experience.lstm_outputs[minibatch["indices"]]
            contrastive_loss, contrastive_metrics = compute_contrastive_loss(
                minibatch, minibatch_lstm, experience.lstm_outputs, trainer_cfg, device
            )

        # Add to total loss:
        loss = (
            pg_loss
            - trainer_cfg.ppo.ent_coef * entropy_loss
            + v_loss * trainer_cfg.ppo.vf_coef
            + l2_init_loss
            + ks_action_loss
            + ks_value_loss
            + trainer_cfg.contrastive.weight * contrastive_loss
        )

        # Update loss tracking:
        for key, value in contrastive_metrics.items():
            setattr(losses, f"{key}_sum", getattr(losses, f"{key}_sum", 0.0) + value)

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
