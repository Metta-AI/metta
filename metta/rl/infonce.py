"""InfoNCE training functionality (contrastive-only policy term).

This implementation assumes the policy exposes LSTM hidden states under the
`"_core_"` key in the minibatch/policy output TensorDict. It does not fall back
to PPO terms. Expand this to support non-LSTM architectures by retrieving the
appropriate representation tensor(s) from the policy outputs.
"""

from typing import Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.rl.trainer_config import TrainerConfig


def compute_infonce_losses(
    minibatch: TensorDict,
    new_logprob: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: TrainerConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
    """Vectorized InfoNCE-style policy loss.

    Notes:
    - This function intentionally does not compute or fall back to PPO losses.
      Value/entropy/approx_kl/clipfrac are returned as zeros for API
      compatibility with PPO-style integrations.
    - Hidden representation is read from `"_core_"` (LSTM). This needs to be
      expanded for non-LSTM architectures.
    """

    # ---------------------------------------------------------------------
    # Placeholder terms for PPO compatibility (not used by InfoNCE)
    # ---------------------------------------------------------------------
    zero = torch.tensor(0.0, device=new_logprob.device, dtype=torch.float32)
    v_loss = zero
    entropy_loss = zero
    approx_kl = zero
    clipfrac = zero

    # ---------------------------------------------------------------------
    # Contrastive InfoNCE over hidden states (assumes LSTM `_core_` outputs)
    # ---------------------------------------------------------------------
    # Currently assumes LSTM `_core_` is present. Needs expansion for non-LSTM architectures.
    if "_core_" not in minibatch.keys():
        raise KeyError("InfoNCE requires hidden representations under '_core_' in the minibatch/policy outputs")
    hidden = minibatch["_core_"]

    # Contrastive hyperparameters from trainer_cfg.contrastive
    K = int(trainer_cfg.contrastive.num_negatives)
    tau = float(trainer_cfg.contrastive.temperature)
    gamma = float(trainer_cfg.contrastive.gamma)
    logsumexp_coef = float(trainer_cfg.contrastive.logsumexp_coef)
    var_reg_coef = float(trainer_cfg.contrastive.var_reg_coef)
    var_reg_target = float(trainer_cfg.contrastive.var_reg_target)

    # Hidden must be [Bseg, T, H]; ensure shape
    if hidden.dim() != 3:
        # If shape is [BT, H], try to infer [B, T, H] from dones
        if "dones" in minibatch and minibatch["dones"].dim() == 2:
            Bseg, T = minibatch["dones"].shape
            H = hidden.shape[-1]
            hidden = hidden.view(Bseg, T, H)
        else:
            raise ValueError("Expected hidden to be [B, T, H] or inferable from 'dones'")

    Bseg, T, H = hidden.shape
    device = hidden.device

    # Primary dones source; expect float in buffer, use bool mask
    if "dones" not in minibatch:
        raise KeyError("'dones' must be present in minibatch for InfoNCE time-offset sampling")
    dones = minibatch["dones"].to(device).bool()  # [Bseg, T]

    # Optional replay pool of hidden states; otherwise use current minibatch
    all_hidden = minibatch.get("all_lstm_hidden", None)
    if all_hidden is None:
        all_hidden = hidden

    # Flattened views
    batch_flat = hidden.reshape(Bseg * T, H)
    all_flat = all_hidden.reshape(-1, H)

    # Build base indices per (segment, timestep)
    seg_idx = torch.arange(Bseg, device=device, dtype=torch.long)
    base_ids = seg_idx[:, None] * T + torch.arange(T, device=device, dtype=torch.long)[None, :]

    # Sample future offsets, clipped by episode boundaries and segment end
    delta = _future_deltas(dones, float(gamma))

    # Compute positive indices and validity
    pos_ids_full = (base_ids + delta).reshape(-1).clamp(max=all_flat.size(0) - 1)
    base_ids_full = base_ids.reshape(-1)
    valid_mask = delta.reshape(-1) >= 1
    if not torch.any(valid_mask):
        # No valid positive pairs within segment boundaries
        metrics = {
            "contrastive_loss": zero,
            "contrastive_infonce": zero,
            "contrastive_logsumexp": zero,
            "contrastive_var_loss": zero,
            "contrastive_pos_sim": zero,
            "contrastive_neg_sim": zero,
            "contrastive_batch_std": zero,
        }
        pg_loss = zero
        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, metrics

    pos_ids = pos_ids_full[valid_mask]
    base_flat_ids = base_ids_full[valid_mask]
    batch_flat = batch_flat[valid_mask]

    eff_B = batch_flat.size(0)
    # Sample negatives; avoid collisions with base/pos indices
    neg_ids = torch.randint(0, all_flat.size(0), (eff_B, K), device=device)
    collisions = (neg_ids == base_flat_ids.unsqueeze(1)) | (neg_ids == pos_ids.unsqueeze(1))
    resample_round = 0
    while torch.any(collisions) and resample_round < 10:
        num_replace = int(collisions.sum().item())
        if num_replace > 0:
            neg_ids[collisions] = torch.randint(0, all_flat.size(0), (num_replace,), device=device)
        collisions = (neg_ids == base_flat_ids.unsqueeze(1)) | (neg_ids == pos_ids.unsqueeze(1))
        resample_round += 1

    # Fetch positive and negative states
    pos_states = all_flat[pos_ids]  # [eff_B, H]
    neg_states = all_flat[neg_ids]  # [eff_B, K, H]

    # Cosine similarities with temperature scaling
    q = torch.nn.functional.normalize(batch_flat, dim=-1)
    kp = torch.nn.functional.normalize(pos_states, dim=-1)
    kn = torch.nn.functional.normalize(neg_states, dim=-1)

    pos_sim = (q * kp).sum(-1, keepdim=True) / float(tau)  # [eff_B, 1]
    neg_sim = (q[:, None, :] * kn).sum(-1) / float(tau)  # [eff_B, K]

    logits = torch.cat((pos_sim, neg_sim), dim=1)  # [eff_B, 1+K]
    targets = torch.zeros(eff_B, dtype=torch.long, device=device)
    infonce = torch.nn.functional.cross_entropy(logits, targets)

    # Optional variance regularizer on q (stop-grad)
    z = q.detach()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(torch.relu(float(var_reg_target) - std))

    # Optional logsumexp regulariser on negatives
    reg = float(logsumexp_coef) * torch.logsumexp(neg_sim, dim=1).mean()

    contrastive_loss = infonce + float(var_reg_coef) * var_loss + reg

    # Use contrastive as policy term
    pg_loss = contrastive_loss

    # Populate metrics for logging
    metrics = {
        "contrastive_loss": contrastive_loss.detach(),
        "contrastive_infonce": infonce.detach(),
        # reg already includes logsumexp_coef
        "contrastive_logsumexp": reg.detach(),
        # raw var regularizer term before coef
        "contrastive_var_loss": var_loss.detach(),
        "contrastive_pos_sim": pos_sim.mean().detach(),
        "contrastive_neg_sim": neg_sim.mean().detach(),
        "contrastive_batch_std": std.mean().detach(),
    }

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac, metrics


@torch.no_grad()
def _future_deltas(
    dones: Tensor,
    gamma: float,
) -> Tensor:
    """Sample future offsets per (segment, t) with boundary clipping.

    - dones: [segments, T] bool mask of episode ends at timestep t
    - gamma: geometric parameter in (0, 1). Larger means shorter expected stride
    Returns: deltas [segments, T] int64 with minimum 1, clipped at boundary
    """
    seg_len = dones.size(1)
    segs = dones.size(0)

    # Raw geometric deltas sampled in a numerically stable way
    u = torch.rand(segs, seg_len, device=dones.device)
    # Geometric with success prob (1 - gamma): delta = floor(log(1-u)/log(gamma))
    # Ensure gamma in (0, 1); if outside, clamp to stable range
    g = float(max(min(gamma, 0.9999), 1e-6))
    deltas = torch.floor(torch.log1p(-u) / torch.log(torch.tensor(g, device=dones.device))).long()
    deltas.clamp_(min=1, max=seg_len - 1)

    # Compute distance to next done or segment end per timestep
    t_idx = torch.arange(seg_len, device=dones.device, dtype=torch.long).expand(segs, seg_len)
    INF = torch.full_like(t_idx, fill_value=seg_len * 2)
    done_pos = torch.where(dones, t_idx, INF)
    suffix_min, _ = torch.cummin(done_pos.flip(1), dim=1)
    next_done_idx = suffix_min.flip(1)
    boundary_idx = torch.where(next_done_idx < INF, next_done_idx, torch.full_like(t_idx, seg_len - 1))
    dist_to_boundary = boundary_idx - t_idx

    # Clip so we never cross an episode boundary or the segment end
    return torch.minimum(deltas, dist_to_boundary)
