import torch
from torch.distributions.utils import logits_to_probs


def get_action_log_prob(log_sftmx_logits, sampled_action):
    """
    Compute log probability of a value given logits.

    Args:
        logits: Unnormalized log probabilities
        value: The specific action selection to compute probability for

    Returns:
        Log probability of the specific action selection
    """
    sampled_action = sampled_action.long().unsqueeze(-1)
    sampled_action, log_pmf = torch.broadcast_tensors(sampled_action, log_sftmx_logits)
    sampled_action = sampled_action[..., :1]
    return log_pmf.gather(-1, sampled_action).squeeze(-1)  # replace w reshape


def entropy(log_sftmx_logits):
    """
    Compute entropy of a categorical distribution given logits.

    Args:
        logits: Unnormalized log probabilities

    Returns:
        Entropy of the distribution
    """
    min_real = torch.finfo(log_sftmx_logits.dtype).min
    log_sftmx_logits = torch.clamp(log_sftmx_logits, min=min_real)
    # logits_to_probs is just softmax. softmax(log(softmax(logits) = softmax(logits)
    p_log_p = log_sftmx_logits * logits_to_probs(log_sftmx_logits)
    return -p_log_p.sum(-1)


def sample_logits(logits: torch.Tensor, action=None):
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
    log_sftmx_logits = logits - logits.logsumexp(dim=-1, keepdim=True)

    if action is None:
        # Sample action if not provided
        # probs shape: [B*T, A]
        probs = logits_to_probs(log_sftmx_logits)
        # Sampled action shape: [B*T, 1], squeeze to [B*T]
        B = logits.shape[0]
        action = torch.multinomial(probs, 1, replacement=True).reshape(B)
        # action = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
    else:
        # Reshape provided action from [B, T, 1] or [B*T] to [B*T]
        action = action.reshape(-1)

    # Ensure action has the expected shape [B*T]
    assert action.shape == logits.shape[:-1], f"Action shape mismatch: expected {logits.shape[:-1]}, got {action.shape}"

    # Calculate log probability for the action selected
    # Shape: [B*T]
    act_logprob = get_action_log_prob(log_sftmx_logits, action)

    # Calculate entropy of the distribution
    # Shape: [B*T]
    logits_entropy = entropy(log_sftmx_logits)

    # Return shapes: [B*T], [B*T], [B*T], [B*T, A]
    return action, act_logprob, logits_entropy, log_sftmx_logits
