from typing import List, Union

import torch
from torch.distributions.utils import logits_to_probs

def log_prob(logits, value):
    """
    Compute log probability of a value given logits.

    Args:
        logits: Unnormalized log probabilities
        value: The value to compute probability for

    Returns:
        Log probability of the value
    """
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    """
    Compute entropy of a categorical distribution given logits.

    Args:
        logits: Unnormalized log probabilities

    Returns:
        Entropy of the distribution
    """
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]], action=None):
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities, either a single tensor or a list of tensors
        action: Optional pre-specified actions to compute probabilities for

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
    """
    normalized_logits = [logit - logit.logsumexp(dim=-1, keepdim=True) for logit in logits]

    if action is None:
        action = torch.stack([torch.multinomial(logits_to_probs(logit), 1).squeeze() for logit in logits])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)

    logprob = torch.stack([log_prob(logit, a) for logit, a in zip(normalized_logits, action, strict=False)]).T.sum(1)
    logits_entropy = torch.stack([entropy(logit) for logit in normalized_logits]).T.sum(1)

    return action.T, logprob, logits_entropy, normalized_logits
