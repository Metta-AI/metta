"""InfoNCE training functionality."""

from typing import Any, Tuple, cast

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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Vectorized InfoNCE-style policy loss with PPO-compatible returns.

    If LSTM hidden states are present in the minibatch (keys like "lstm_hidden", "hidden",
    or a component name that produced a [B, T, H] tensor), we compute a contrastive
    InfoNCE objective by sampling positives within the same segment and negatives from
    the available hidden pool (fallbacks to in-minibatch if a replay pool isn't provided).

    Otherwise, this falls back to an unclipped policy gradient term: (-adv * new_logprob).mean().
    The value-loss, entropy, approx_kl and clipfrac match PPO for logging/compatibility.
    """

    # ---------------------------------------------------------------------
    # Value and entropy terms (same as PPO for compatibility)
    # ---------------------------------------------------------------------
    returns = minibatch["returns"]
    old_values = minibatch["values"]
    newvalue_reshaped = newvalue.view(returns.shape)

    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue_reshaped - returns) ** 2
        vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
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

    with torch.no_grad():
        logratio = new_logprob - minibatch["act_log_prob"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    # ---------------------------------------------------------------------
    # Contrastive InfoNCE over hidden states if available; else fallback
    # ---------------------------------------------------------------------
    hidden = _locate_hidden_tensor(minibatch)
    if hidden is None:
        # Fallback: basic InfoNCE-style policy term on action logprob
        pg_loss = (-adv * new_logprob).mean()
        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

    # Read optional contrastive hyperparameters from trainer_cfg if present
    K = cast(int, _safe_get(trainer_cfg, ["contrastive", "num_negatives"], default=64))
    tau = cast(float, _safe_get(trainer_cfg, ["contrastive", "temperature"], default=0.1))
    gamma = cast(float, _safe_get(trainer_cfg, ["contrastive", "gamma"], default=0.95))
    logsumexp_coef = cast(float, _safe_get(trainer_cfg, ["contrastive", "logsumexp_coef"], default=0.0))
    var_reg_coef = cast(float, _safe_get(trainer_cfg, ["contrastive", "var_reg_coef"], default=0.0))
    var_reg_target = cast(float, _safe_get(trainer_cfg, ["contrastive", "var_reg_target"], default=1.0))

    # Hidden must be [Bseg, T, H]; ensure shape
    if hidden.dim() != 3:
        # If shape is [BT, H], try to infer B and T from dones
        if "dones" in minibatch and minibatch["dones"].dim() == 2:
            Bseg, T = minibatch["dones"].shape
            H = hidden.shape[-1]
            hidden = hidden.view(Bseg, T, H)
        else:
            # As a conservative fallback, skip contrastive term
            pg_loss = (-adv * new_logprob).mean()
            return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

    Bseg, T, H = hidden.shape
    device = hidden.device

    # Primary dones source; expect float in buffer, use bool mask
    if "dones" not in minibatch:
        pg_loss = (-adv * new_logprob).mean()
        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
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
        pg_loss = (-adv * new_logprob).mean()
        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

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

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


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


def _safe_get(cfg: object, path: list[str], default) -> Any:
    cur = cfg
    for key in path:
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur


def _locate_hidden_tensor(minibatch: TensorDict) -> Tensor | None:
    """Best-effort retrieval of a [B, T, H] hidden tensor from minibatch.

    Tries common keys. If none found, returns None.
    """
    candidate_keys = [
        "lstm_hidden",
        "hidden",
        "_core_",  # typical LSTM component name in ComponentPolicy
        "core",
        "hidden_states",
        "hidden_state",
        "latent",
        "encoder",
    ]
    for key in candidate_keys:
        if key in minibatch.keys() and isinstance(minibatch.get(key), Tensor):
            tensor = minibatch.get(key)
            if tensor is not None and tensor.dim() in (2, 3):
                return tensor
    return None


# SAMPLE CODE
# @torch.no_grad()
# def _future_deltas(
#     dones: Tensor,  # [segments, T] bool
#     gamma: float,  # geometric parameter
# ) -> Tensor:  # [segments, T] int64
#     """
#     Geometrically-distributed offsets, clipped so we never cross an episode boundary
#     (or the end of the segment). Fully vectorised on GPU.
#     """
#     seg_len = dones.size(1)
#     segs = dones.size(0)

#     # Sample raw geometric deltas
#     u = torch.rand(segs, seg_len, device=dones.device)
#     deltas = torch.floor(torch.log1p(-u) / torch.log(torch.tensor(gamma, device=dones.device))).long()
#     deltas.clamp_(min=1, max=seg_len - 1)

#     # Compute distance to the nearest future 'done' for every timestep.
#     # Example: dones=[F F T F] → distances=[2, 1, 0, 0]
#     # If there is no future 'done', allow moving up to the segment end: distances = (seg_len-1 - t)
#     t_idx = torch.arange(seg_len, device=dones.device, dtype=torch.long).expand(segs, seg_len)
#     INF = torch.full_like(t_idx, fill_value=seg_len * 2)
#     done_pos = torch.where(dones, t_idx, INF)
#     suffix_min, _ = torch.cummin(done_pos.flip(1), dim=1)
#     next_done_idx = suffix_min.flip(1)
#     boundary_idx = torch.where(next_done_idx < INF, next_done_idx, torch.full_like(t_idx, seg_len - 1))
#     dist_to_boundary = boundary_idx - t_idx

#     # Clip deltas so we never step past a 'done' or segment end
#     deltas = torch.minimum(deltas, dist_to_boundary)

#     return deltas  # [segments, T]


# def compute_contrastive_loss_fast(
#     minibatch: Dict[str, Tensor],
#     lstm_hidden: Tensor,  # [Bseg, T, H]
#     all_lstm_hidden: Tensor,  # [S, T, H]  (huge replay window on-device)
#     trainer_cfg: Any,
#     device: torch.device,
# ) -> Tuple[Tensor, Dict[str, float]]:
#     """Fully-vectorised InfoNCE + log-sum-exp regulariser."""

#     Bseg, T, H = lstm_hidden.shape
#     B = Bseg * T
#     K = trainer_cfg.contrastive.num_negatives
#     tau = trainer_cfg.contrastive.temperature
#     gamma = trainer_cfg.contrastive.gamma
#     logsumexp_coef = trainer_cfg.contrastive.logsumexp_coef

#     # flatten views
#     batch_flat = lstm_hidden.reshape(B, H)  # current states
#     all_flat = all_lstm_hidden.reshape(-1, H)  # replay window

#     # positive indices
#     seg_idx = minibatch["indices"].to(device, dtype=torch.long)  # [Bseg]
#     base_ids = seg_idx[:, None] * T + torch.arange(T, device=device, dtype=torch.long)  # [Bseg, T]
#     # Correct distance-to-boundary clipping (no crossing 'done' or segment end)
#     delta = _future_deltas(minibatch["dones"].to(device), gamma)  # [Bseg, T]
#     pos_ids_full = (base_ids + delta).reshape(-1).clamp(max=all_flat.size(0) - 1).to(device, dtype=torch.long)
#     base_ids_full = base_ids.reshape(-1)
#     valid_mask = delta.reshape(-1) >= 1
#     # If no valid positions, return zero loss to avoid NaNs
#     if not torch.any(valid_mask):
#         zero = torch.tensor(0.0, device=device, dtype=torch.float32)
#         metrics = {
#             "contrastive_loss": 0.0,
#             "contrastive_infonce": 0.0,
#             "contrastive_logsumexp": 0.0,
#             "contrastive_pos_sim": 0.0,
#             "contrastive_neg_sim": 0.0,
#             "contrastive_var_loss": 0.0,
#             "contrastive_batch_std": 0.0,
#         }
#         return zero, metrics
#     pos_ids = pos_ids_full[valid_mask]
#     base_flat_ids = base_ids_full[valid_mask]
#     batch_flat = batch_flat[valid_mask]

#     # negative indices
#     # Sample K negatives that are NOT equal to current or positive indices
#     eff_B = batch_flat.size(0)
#     neg_ids = torch.randint(0, all_flat.size(0), (eff_B, K), device=device)
#     collisions = (neg_ids == base_flat_ids.unsqueeze(1)) | (neg_ids == pos_ids.unsqueeze(1))
#     resample_round = 0
#     # Re-sample colliding slots until none remain (expected to converge quickly)
#     while torch.any(collisions) and resample_round < 10:
#         num_replace = int(collisions.sum().item())
#         if num_replace > 0:
#             neg_ids[collisions] = torch.randint(0, all_flat.size(0), (num_replace,), device=device)
#         collisions = (neg_ids == base_flat_ids.unsqueeze(1)) | (neg_ids == pos_ids.unsqueeze(1))
#         resample_round += 1

#     # fetch states
#     pos_states = all_flat[pos_ids]  # [eff_B, H]
#     neg_states = all_flat[neg_ids]  # [eff_B, K, H]

#     # cosine similarities
#     q = torch.nn.functional.normalize(batch_flat, dim=-1)
#     kp = torch.nn.functional.normalize(pos_states, dim=-1)
#     kn = torch.nn.functional.normalize(neg_states, dim=-1)

#     pos_sim = (q * kp).sum(-1, keepdim=True) / tau  # [eff_B, 1]
#     neg_sim = (q[:, None, :] * kn).sum(-1) / tau  # [eff_B, K]

#     print("I AM COMPUTING CONTRASTIVE LOSS PROBABLY")

#     # infonce
#     logits = torch.cat((pos_sim, neg_sim), dim=1)  # [eff_B, 1+K]
#     targets = torch.zeros(eff_B, dtype=torch.long, device=device)
#     infonce = torch.nn.functional.cross_entropy(logits, targets)

#     var_target = trainer_cfg.contrastive.var_reg_target
#     z = q.detach()  # (eff_B, H)  stop-grad to avoid wasting mem
#     std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-04)

#     var_loss = torch.mean(torch.relu(var_target - std))

#     # extra regulariser
#     reg = logsumexp_coef * torch.logsumexp(neg_sim, dim=1).mean()

#     loss = infonce + reg + trainer_cfg.contrastive.var_reg_coef * var_loss

#     # metrics
#     metrics = {
#         "contrastive_loss": loss.item(),
#         "contrastive_infonce": infonce.item(),
#         "contrastive_logsumexp": reg.item(),
#         "contrastive_pos_sim": pos_sim.mean().item(),
#         "contrastive_neg_sim": neg_sim.mean().item(),
#         "contrastive_var_loss": var_loss.item(),
#         "contrastive_batch_std": std.mean().item(),
#     }
#     return loss, metrics
