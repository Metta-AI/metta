import torch
from typing import List, Union
from torch.distributions.utils import logits_to_probs


def log_prob(logits, value):
    """
    Compute log probability of a value given logits.
    
    Args:
        logits: Unnormalized log probabilities
        value: The specific action selection to compute probability for
        
    Returns:
        Log probability of the specific action selection
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


def sample_logits(logits: torch.Tensor,
        action=None):
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities of shape [batch_size * bptt, num_actions].
        action: Optional pre-specified actions of shape [batch_size, bptt, 1] or [batch_size * bptt].
                If provided, log probabilities and entropy for these actions are computed.

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
        Shapes: [B*T], [B*T], [B*T], [B*T, A] respectively.
    """
    # Input logits shape: [B*T, A]
    # Input action shape (if provided): [B, T, 1]

    # Normalize logits for numerical stability
    # Shape: [B*T, A]
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)

    if action is None:
        # Sample action if not provided
        # probs shape: [B*T, A]
        probs = logits_to_probs(normalized_logits)
        # Sampled action shape: [B*T, 1], squeeze to [B*T]
        action = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
    else:
        # Reshape provided action from [B, T, 1] or [B*T] to [B*T]
        action = action.reshape(-1) 

    # Ensure action has the expected shape [B*T]
    assert action.shape == logits.shape[:-1], f"Action shape mismatch: expected {logits.shape[:-1]}, got {action.shape}"

    # Calculate log probability for the action
    # Shape: [B*T]
    logprob = log_prob(normalized_logits, action)

    # Calculate entropy of the distribution
    # Shape: [B*T]
    logits_entropy = entropy(normalized_logits)

    # Return shapes: [B*T], [B*T], [B*T], [B*T, A]
    return action, logprob, logits_entropy, normalized_logits


# def log_prob(logits, value):
#     """
#     Compute log probability of a value given logits.
    
#     Args:
#         logits: Unnormalized log probabilities
#         value: The specific action selection to compute probability for
        
#     Returns:
#         Log probability of the specific action selection
#     """
#     value = value.long().unsqueeze(-1)
#     value, log_pmf = torch.broadcast_tensors(value, logits)
#     value = value[..., :1]
#     return log_pmf.gather(-1, value).squeeze(-1)


# def entropy(logits):
#     """
#     Compute entropy of a categorical distribution given logits.
    
#     Args:
#         logits: Unnormalized log probabilities
        
#     Returns:
#         Entropy of the distribution
#     """
#     min_real = torch.finfo(logits.dtype).min
#     logits = torch.clamp(logits, min=min_real)
#     p_log_p = logits * logits_to_probs(logits)
#     return -p_log_p.sum(-1)


# def sample_logits(logits: torch.Tensor,
#         action=None):
#     # is_discrete = False
#     # if isinstance(logits, torch.Tensor):
#     #     is_discrete = True
#     logits = logits.unsqueeze(0) # do we need to unsqueeze?
#     # else: #multi-discrete
#     #     logits = torch.nn.utils.rnn.pad_sequence(
#     #         [l.transpose(0,1) for l in logits], 
#     #         batch_first=False, 
#     #         padding_value=-torch.inf
#     #     ).permute(1,2,0)

#     normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
#     probs = logits_to_probs(normalized_logits)

#     if action is None:
#         action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True)
#         action = action.reshape(probs.shape[:-1])
#     else:
#         batch = logits[0].shape[0]
#         action = action.view(batch, -1).T
#         probs = logits_to_probs(normalized_logits)

#     assert len(logits) == len(action)
#     logprob = log_prob(normalized_logits, action)
#     logits_entropy = entropy(normalized_logits).sum(0)

#     # if is_discrete:
#     return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0), normalized_logits

#     # return action.T, logprob, logits_entropy, normalized_logits

