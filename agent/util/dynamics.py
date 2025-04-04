import torch
import numpy as np
from scipy import stats


def analyze_sv(S: torch.Tensor): 
    # the type hint can be changed but we'd have to update how the outputs flow into metta_layer.py and through logging in trainer.py -- 
    # Lars: we might want one of them to be a string at somepoint, to output qualitative metrics e.g. "critical" or "sub-critical"
    ''' Lars, go to town! S is the tensor of singular values. I wrote an example of how to use this function. 
    Right now, it supports outputting two metrics but we can obv add more.
    '''
    # eigens = S ** 2
    # sorted_eigens = torch.sort(eigens, descending=True).values
    # largest_eigen = sorted_eigens[0].item()
    # smallest_eigen = sorted_eigens[-1].item()
    
    sorted_sv = torch.sort(S, descending=True).values
    largest_sv = sorted_sv[0].item()
    smallest_sv = sorted_sv[-1].item()
    non_zero_sv = sorted_sv[sorted_sv > 0]
    smallest_non_zero_sv = non_zero_sv[-1].item()
    sv_ratio = largest_sv / smallest_non_zero_sv # also known as the condition number   

    assert len(sorted_sv) > 5, "Need at least 6 singular values for meaningful fit"

    # Linear fit in log-log space to check for power law (indicator of criticality)
    log_indices = np.log(np.arange(1, len(sorted_sv) + 1))
    log_sv = torch.log(sorted_sv + 1e-10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_indices, log_sv)

    # TODO:  make the regime determination more sophisticated
    # Determine regime
    if sv_ratio > 1000:  # Very high ratio indicates subcritical
        regime = 0 # "Subcritical"
    elif r_value**2 > 0.9 and -1.5 < slope < -0.5:
                
        regime = 1 # "Critical" -- a ood power-law fit with reasonable exponent indicates criticality
    else:
        regime = 2 # "Chaotic"

    return sv_ratio, r_value**2, slope, largest_sv
