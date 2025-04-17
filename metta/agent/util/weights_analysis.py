import torch
from scipy import stats


def analyze_weights(weights: torch.Tensor, delta: float = 0.01) -> dict:
    """Analyze weight matrix properties including singular values, effective rank,
    weight norms and other dynamics-related metrics.

    Args:
        weights: Weight tensor to analyze
        delta: Threshold for effective rank calculation (default: 0.01)

    Returns:
        Dictionary containing all computed metrics including:
        - Singular value statistics (ratio, largest, mean, collapsed dimensions)
        - Power law fit metrics (R² value, slope)
        - Weight norm
        - Effective rank
        - Basic statistics if SVD fails
    """
    metrics = {}

    # Basic weight statistics
    weight_norm = torch.linalg.norm(weights).item()
    metrics["weight_norm"] = weight_norm

    try:
        # Add small noise to prevent convergence issues
        noise = torch.randn_like(weights) * 1e-7
        _, S, _ = torch.linalg.svd(weights + noise)

        # Analyze singular values
        sorted_sv = torch.sort(S, descending=True).values
        sorted_sv_non_zero = sorted_sv[sorted_sv > 1e-10]  # ignore singular values close to zero

        # Basic SV metrics
        metrics.update(
            {
                "mean_sv": torch.mean(S).item(),
                "largest_sv": sorted_sv[0].item(),
                "smallest_sv": sorted_sv_non_zero[-1].item(),
                "collapsed_dim_%": (len(sorted_sv) - len(sorted_sv_non_zero)) / len(sorted_sv),
                "sv_ratio": sorted_sv[0].item() / sorted_sv_non_zero[-1].item(),  # condition number
            }
        )

        # Power law fit analysis (indicator of criticality)
        if len(sorted_sv) > 5:
            log_indices = torch.log(torch.arange(1, len(sorted_sv_non_zero) + 1, device=S.device).float())
            log_sv = torch.log(sorted_sv_non_zero + 1e-10)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_indices.cpu().numpy(), log_sv.cpu().numpy()
            )

            metrics.update({"sv_loglog_r2": r_value**2, "sv_loglog_slope": slope})

        # Calculate effective rank
        if len(S) > 1:
            total_sum = S.sum()
            cumulative_sum = torch.cumsum(S, dim=0)
            threshold = (1 - delta) * total_sum
            effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1
            metrics["effective_rank"] = effective_rank

    except Exception as e:
        # Fallback metrics if SVD fails
        metrics.update({"svd_error": str(e), "avg_abs_weight": torch.abs(weights).mean().item()})
        try:
            metrics["condition_number"] = torch.linalg.cond(weights).item()
        except RuntimeError:
            metrics["condition_number"] = float("nan")

    return metrics
