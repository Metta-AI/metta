import torch
import numpy as np
from scipy import stats


def analyze_sv(S: torch.Tensor): 
    sorted_sv = torch.sort(S, descending=True).values
    sorted_sv_non_zero = sorted_sv[sorted_sv > 1e-10] # ignore singular values close to zero as they do not contribute to the dynamics
    mean_sv = torch.mean(S).item()
    largest_sv = sorted_sv[0].item()
    smallest_sv = sorted_sv_non_zero[-1].item()
    collapsed_dim_ratio = (len(sorted_sv) - len(sorted_sv_non_zero)) / len(sorted_sv)

    sv_ratio = largest_sv / smallest_sv # also known as the condition number   

    if len(sorted_sv) <= 5:
        print(f"Not enough singular values to fit a power law. Only {len(sorted_sv)} singular values.")
        return {}
        
    # Linear fit in log-log space to check for power law (indicator of criticality)
    log_indices_np = torch.log(torch.arange(1, len(sorted_sv) + 1, device=S.device).float()).cpu().numpy()
    log_sv_np = torch.log(sorted_sv + 1e-10).cpu().numpy()

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_indices_np, log_sv_np)

    metrics = {
        'sv_ratio': sv_ratio,
        'r2_value': r_value**2,
        'slope': slope,
        'largest_sv': largest_sv,
        'mean_sv': mean_sv,
        'collapsed_dim_%': collapsed_dim_ratio,
    }
    
    return metrics
