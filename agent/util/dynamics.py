import torch
import numpy as np
from scipy import stats


def analyze_sv(S: torch.Tensor): 
    # the type hint can be changed but we'd have to update how the outputs flow into metta_layer.py and through logging in trainer.py -- 
    # Lars: we might want one of them to be a string at somepoint, to output qualitative metrics e.g. "critical" or "sub-critical"
    ''' Lars, go to town! S is the tensor of singular values. I wrote an example of how to use this function. 
    Right now, it supports outputting two metrics but we can obv add more.
    '''
    
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
