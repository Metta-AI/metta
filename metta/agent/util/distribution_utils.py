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
    # Convert value to long type (integer) and add an extra dimension at the end
    # Example: [1, 2] -> [[1], [2]]
    value = value.long().unsqueeze(-1)

    # Broadcast both tensors to a compatible shape
    # Note: Despite the variable name "log_pmf", this does NOT normalize the logits
    # It simply renames the logits tensor and ensures compatible dimensions
    # If logits is [batch_size, vocab_size] and value is [batch_size, 1],
    # both will be broadcasted to have the same batch dimensions
    value, log_pmf = torch.broadcast_tensors(value, logits)

    # Keep only the first element of the last dimension of the value tensor
    # This ensures the indices have the right shape for the gather operation
    # Example: If broadcasted value is [batch_size, vocab_size, 1],
    # this keeps only [batch_size, 1]
    value = value[..., :1]

    # Use the values in the 'value' tensor as indices to gather elements from 'log_pmf'
    # along the last dimension (-1), then remove the last dimension with squeeze
    # This effectively selects the logit values at the positions specified by 'value'
    # IMPORTANT: This does NOT compute actual log probabilities - it simply returns
    # the raw logit values at the specified indices
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    """
    Compute entropy of a categorical distribution given logits.

    Args:
        logits: Unnormalized log probabilities

    Returns:
        Entropy of the distribution
    """
    # Get the minimum representable value for the data type (to prevent -inf values)
    min_real = torch.finfo(logits.dtype).min

    # Clamp logits to avoid numerical instability (prevents values lower than min_real)
    # This prevents -inf values that can cause NaN results in subsequent calculations
    logits = torch.clamp(logits, min=min_real)

    # Convert logits to probabilities using softmax (through logits_to_probs function)
    # Then multiply by the original logits
    # NOTE: This is NOT the standard formula for entropy calculation!
    # Standard entropy would be: -sum(p * log(p)) where p = softmax(logits)
    # This implementation uses: -sum(logits * softmax(logits))
    # These are not mathematically equivalent
    p_log_p = logits * logits_to_probs(logits)

    # Sum the values along the last dimension and negate the result
    # For uniform distributions, this approximates log(n) where n is the number of categories
    # For deterministic distributions, this returns approximately -logit_max
    # (the negative of the maximum logit value)
    return -p_log_p.sum(-1)


def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]], action=None, verbose=False):
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities, either a single tensor or a list of tensors
        action: Optional pre-specified actions to compute probabilities for

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
    """
    # Normalize each logits tensor by subtracting the logsumexp (log of sum of exponentials)
    # This converts raw logits to log probabilities (normalization in log space)
    # Converts a list of tensors of shape [batch_size, vocab_size] to a list of normalized tensors
    normalized_logits = [logit - logit.logsumexp(dim=-1, keepdim=True) for logit in logits]

    # Debug output if verbose mode is enabled
    if verbose:
        print(f"logits has len {len(logits)}")

    # If no actions are provided, sample actions based on the probabilities
    if action is None:
        # More detailed debug output in verbose mode
        if verbose:
            lgt = logits[0]
            print(f"logits[0] {lgt}, has shape {lgt.shape}")
            lgt_prob = logits_to_probs(lgt)  # Convert logits to probabilities using softmax
            print(f"logits_to_probs(logit): {lgt_prob}")
            torchmn = torch.multinomial(lgt_prob, 1)  # Sample one index per row based on probabilities
            print(f"torch.multinomial(logits_to_probs(logit), 1): {torchmn}")
            squeeze = torchmn.squeeze()  # Remove singular dimensions
            print(f"torch.multinomial(logits_to_probs(logit), 1).squeeze(): {squeeze}")

        # Sample actions from each logits tensor
        # For each tensor in the logits list:
        # 1. Convert to probabilities using softmax (logits_to_probs)
        # 2. Sample one index per row using multinomial
        # 3. Remove singular dimensions with squeeze
        # 4. Stack all results
        # This results in a tensor of shape [num_logits_tensors, batch_size]
        action = torch.stack([torch.multinomial(logits_to_probs(logit), 1).squeeze() for logit in logits])

        if verbose:
            print(f"action has shape {action.shape}")
    else:
        # If actions are provided, reshape them
        # Get the batch size from the first logits tensor
        batch = logits[0].shape[0]

        # Reshape the action tensor from [batch_size * num_logits_tensors] to [num_logits_tensors, batch_size]
        # This assumes that actions are provided in a flattened format where the first batch_size
        # elements correspond to the first logits tensor, the next batch_size to the second, and so on
        action = action.view(batch, -1).T

    # Verify that the number of action tensors matches the number of logits tensors
    assert len(logits) == len(action)

    # Compute log probabilities for each (logit, action) pair
    # NOTE: This uses the log_prob function which, as we discovered, doesn't actually
    # compute log probabilities but rather just selects the logit values at the specified indices
    # The results are stacked, transposed, and summed along dimension 1
    # This gives a tensor of shape [batch_size] containing the sum of selected logit values
    logprob = torch.stack([log_prob(logit, a) for logit, a in zip(normalized_logits, action, strict=False)]).T.sum(1)

    # Compute entropy for each logits tensor
    # As we discovered, this doesn't compute standard Shannon entropy but rather a different formula
    # The results are stacked, transposed, and summed along dimension 1
    # This gives a tensor of shape [batch_size] containing the sum of entropies
    logits_entropy = torch.stack([entropy(logit) for logit in normalized_logits]).T.sum(1)

    # Return a tuple of:
    # 1. Actions (transposed back to shape [batch_size, num_logits_tensors])
    # 2. Log probabilities (which are actually just sums of selected logit values)
    # 3. Entropies (which use a non-standard entropy formula)
    # 4. Normalized logits (the log probabilities for each logits tensor)
    return action.T, logprob, logits_entropy, normalized_logits


def faster_sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]], action=None, verbose=False):
    """
    Sample actions from logits and compute log probabilities and entropy.
    Optimized for speed.

    Args:
    logits: Unnormalized log probabilities, either a single tensor or a list of tensors
    action: Optional pre-specified actions to compute probabilities for
    Returns:
    Tuple of (action, log_probability, entropy, normalized_logits)
    """
    # Convert single tensor to list for consistent handling
    if isinstance(logits, torch.Tensor):
        logits = [logits]

    # Fast normalization - subtract logsumexp once per logit tensor
    normalized_logits = []
    for logit in logits:
        # In-place subtraction when possible is faster
        normalized_logit = logit - logit.logsumexp(dim=-1, keepdim=True)
        normalized_logits.append(normalized_logit)

    batch_size = normalized_logits[0].shape[0]
    num_logits = len(normalized_logits)

    if action is None:
        # Pre-allocate action tensor with correct shape
        device = normalized_logits[0].device
        action = torch.empty((batch_size, num_logits), dtype=torch.long, device=device)

        # Debug output if requested
        if verbose:
            print(f"logits has len {num_logits}")
            lgt = logits[0]
            print(f"logits[0] {lgt}, has shape {lgt.shape}")
            # Faster exp calculation directly on normalized logits
            lgt_prob = torch.exp(normalized_logits[0])
            print(f"logits_to_probs(logit): {lgt_prob}")

        # Faster sampling directly into pre-allocated tensor
        for i, logit in enumerate(normalized_logits):
            # Direct exp is faster than calling separate function
            probs = torch.exp(logit)
            # Directly sample and place in pre-allocated tensor
            action[:, i] = torch.multinomial(probs, 1).flatten()

        if verbose:
            print(f"action has shape {action.shape}")
    else:
        # Fast reshape of provided action if needed
        if action.dim() == 1:
            if len(logits) == 1:
                # Single logit case
                action = action.view(batch_size, 1)
            else:
                # Multiple logits case
                action = action.view(batch_size, -1)

    # Fast computation of log probabilities and entropy
    # Pre-allocate tensors
    logprob = torch.zeros(batch_size, device=normalized_logits[0].device)
    logits_entropy = torch.zeros(batch_size, device=normalized_logits[0].device)

    # Loop with direct computations instead of function calls
    for i, logit in enumerate(normalized_logits):
        # Extract actions for this logit
        act_i = action[:, i].unsqueeze(-1)

        # Directly gather log probs - faster than separate function
        logprob_i = torch.gather(logit, -1, act_i).squeeze(-1)
        logprob.add_(logprob_i)  # In-place addition is faster

        # Direct entropy calculation - faster than separate function
        probs = torch.exp(logit)
        entropy_i = -torch.sum(probs * logit, dim=-1)
        logits_entropy.add_(entropy_i)  # In-place addition is faster

    return action, logprob, logits_entropy, normalized_logits
