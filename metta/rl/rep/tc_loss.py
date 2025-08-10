import torch
import torch.nn.functional as F


def cosine_tc(preds: torch.Tensor, tgts: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Temporal consistency cosine loss with discounting."""

    preds = F.normalize(preds, dim=-1)
    tgts = F.normalize(tgts, dim=-1)
    cos = (preds * tgts).sum(dim=-1)
    k = preds.size(1)
    disc = (gamma ** torch.arange(k, device=preds.device)).view(1, k)
    return ((1.0 - cos) * disc).mean()
