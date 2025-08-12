"""Loss computation functions for PPO training."""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.experience import Experience
from metta.rl.modules import DynamicsModel, ProjectionHead
from metta.rl.trainer_config import RepresentationLearningConfig, TrainerConfig

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


def get_loss_experience_spec(nvec: list[int] | torch.Tensor, act_dtype: torch.dtype) -> Composite:
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


def process_minibatch_update(
    policy: PolicyAgent,
    experience: Experience,
    minibatch: TensorDict,
    policy_td: TensorDict,
    trainer_cfg: TrainerConfig,
    indices: Tensor,
    prio_weights: Tensor,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    policy_td = policy(policy_td, action=minibatch["actions"])

    old_act_log_prob = minibatch["act_log_prob"]
    new_logprob = policy_td["act_log_prob"].reshape(old_act_log_prob.shape)
    entropy = policy_td["entropy"]
    newvalue = policy_td["value"]
    full_logprobs = policy_td["full_log_probs"]

    logratio = new_logprob - old_act_log_prob
    importance_sampling_ratio = logratio.exp()

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
    adv = prio_weights * adv

    from metta.rl.ppo import compute_ppo_losses

    # Compute losses
    pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
        minibatch,
        new_logprob,
        entropy,
        newvalue,
        importance_sampling_ratio,
        adv,
        trainer_cfg,
    )

    # Kickstarter losses
    ks_action_loss, ks_value_loss = kickstarter.loss(
        agent_step,
        full_logprobs,
        newvalue,
        policy_td["env_obs"],
    )

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

    # Update values and ratio in experience buffer
    update_td = TensorDict(
        {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
        batch_size=minibatch.batch_size,
    )
    experience.update(indices, update_td)

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
    losses.current_logprobs_sum += new_logprob.mean().item()

    return loss


class RepresentationLoss(nn.Module):
    """Representation learning loss combining contrastive, temporal consistency and prediction."""

    def __init__(
        self,
        projection_head: ProjectionHead,
        dynamics_model: DynamicsModel,
        cfg: "RepresentationLearningConfig",
    ) -> None:
        super().__init__()
        self.projection_head = projection_head
        self.dynamics_model = dynamics_model
        self.cfg = cfg

    def sample_positive_offsets(self, T: int, device: torch.device) -> torch.Tensor:
        """Sample positive offsets for each timestep using geometric distribution."""
        if T <= 1:
            return torch.zeros(0, dtype=torch.long, device=device)
        dist = torch.distributions.Geometric(probs=1 - self.cfg.alpha)
        k = dist.sample((T - 1,)).to(device).clamp(min=1)
        max_range = torch.arange(1, T, device=device)
        k = torch.minimum(k, max_range)
        return k.to(torch.long)

    def compute_contrastive(self, z: Tensor, mask: Tensor) -> Tensor:
        """Compute InfoNCE contrastive loss."""
        T, B, D = z.shape
        device = z.device
        offsets = self.sample_positive_offsets(T, device)
        anchors: list[Tensor] = []
        positives: list[Tensor] = []
        negatives: list[Tensor] = []
        for b in range(B):
            for t in range(T - 1):
                if not (mask[t, b] and mask[min(t + offsets[t].item(), T - 1), b]):
                    continue
                t_pos = min(t + offsets[t].item(), T - 1)
                anchor = z[t, b]
                pos = z[t_pos, b]
                neg_choices = [i for i in range(T) if i not in (t, t_pos) and mask[i, b]]
                if len(neg_choices) == 0:
                    continue
                neg_idx = torch.tensor(neg_choices, device=device, dtype=torch.long)
                if len(neg_idx) < self.cfg.num_negatives:
                    choice = torch.randint(0, len(neg_idx), (self.cfg.num_negatives,), device=device)
                    neg_idx = neg_idx[choice]
                else:
                    perm = torch.randperm(len(neg_idx), device=device)[: self.cfg.num_negatives]
                    neg_idx = neg_idx[perm]
                negatives.append(z[neg_idx, b])
                anchors.append(anchor)
                positives.append(pos)
        if not anchors:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        anchor_t = self.projection_head(torch.stack(anchors))
        pos_t = self.projection_head(torch.stack(positives))
        neg_t = self.projection_head(torch.stack(negatives))
        anchor_t = F.normalize(anchor_t, dim=-1)
        pos_t = F.normalize(pos_t, dim=-1)
        neg_t = F.normalize(neg_t, dim=-1)
        pos_sim = (anchor_t * pos_t).sum(-1, keepdim=True) / self.cfg.tau
        neg_sim = torch.einsum("nd,nkd->nk", anchor_t, neg_t) / self.cfg.tau
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def compute_tc(self, z: Tensor, mask: Tensor) -> Tensor:
        z_t = z[:-1]
        z_tp1 = z[1:]
        valid = mask[:-1] & mask[1:]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)
        if self.cfg.loss_tc_type == "l2":
            diff = (z_t - z_tp1).norm(dim=-1)
        else:
            diff = 1 - F.cosine_similarity(z_t, z_tp1, dim=-1)
        return (diff * valid).sum() / valid.sum()

    def compute_pred(self, z: Tensor, a: Tensor, mask: Tensor) -> Tensor:
        z_t = z[:-1]
        z_tp1 = z[1:].detach()
        a_t = a[:-1]
        valid = mask[:-1] & mask[1:]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device, dtype=torch.float32)
        pred = self.dynamics_model(z_t, a_t)
        if self.cfg.loss_pred_type == "l2":
            diff = (pred - z_tp1).norm(dim=-1)
        else:
            diff = 1 - F.cosine_similarity(pred, z_tp1, dim=-1)
        return (diff * valid).sum() / valid.sum()

    def forward(self, z: Tensor, a: Tensor, mask: Tensor) -> dict[str, Tensor]:
        contrast = self.compute_contrastive(z, mask)
        tc = self.compute_tc(z, mask)
        pred = self.compute_pred(z, a, mask)
        total = self.cfg.lambda_contrast * contrast + self.cfg.lambda_tc * tc + self.cfg.lambda_pred * pred
        return {
            "contrast": contrast,
            "tc": tc,
            "pred": pred,
            "total": total,
        }
