import torch
import numpy as np
from scipy import stats


def analyze_sv(S: torch.Tensor): 
    sorted_sv = torch.sort(S, descending=True).values
    mean_sv = torch.mean(S).item()
    largest_sv = sorted_sv[0].item()
    smallest_sv = sorted_sv[-1].item()
    non_zero_sv = sorted_sv[sorted_sv > 0]
    smallest_non_zero_sv = non_zero_sv[-1].item()
    sv_ratio = largest_sv / smallest_non_zero_sv # also known as the condition number   

    if len(sorted_sv) <= 5:
        print(f"Not enough singular values to fit a power law. Only {len(sorted_sv)} singular values.")
        return {}
    
    #TODO: compare R2 values when using all the singular values vs. only the non-zero ones
    
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
    }
    
    return metrics
