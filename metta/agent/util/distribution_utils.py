from typing import List, Optional, Tuple, Union

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.utils import logits_to_probs


def log_prob(normalized_logits, value):
    """
    Compute log probability of a value given logits.

    Args:
        logits: Unnormalized log probabilities
        value: The value to compute probability for

    Returns:
        Log probability of the value
    """
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, normalized_logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(normalized_logits):
    """
    Compute entropy of a categorical distribution given logits.

    Args:
        logits: Unnormalized log probabilities

    Returns:
        Entropy of the distribution
    """
    min_real = torch.finfo(normalized_logits.dtype).min
    logits = torch.clamp(normalized_logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits_old(logits: Union[torch.Tensor, List[torch.Tensor]], action=None):
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities, either a single tensor or a list of tensors
        action: Optional pre-specified actions to compute probabilities for

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
    """
    normalized_logits = [logit - logit.logsumexp(dim=-1, keepdim=True) for logit in logits]

    B = logits[0].shape[0]
    if action is None:
        action = torch.stack([torch.multinomial(logits_to_probs(logit), 1).reshape(B) for logit in logits])
    else:
        action = action.view(B, -1).T

    assert len(logits) == len(action)

    logprob = torch.stack([log_prob(logit, a) for logit, a in zip(normalized_logits, action, strict=False)]).T.sum(1)
    logits_entropy = torch.stack([entropy(logit) for logit in normalized_logits]).T.sum(1)

    return action.T, logprob, logits_entropy, normalized_logits


@torch.jit.script
def sample_logits(
    logits: Tensor | List[Tensor], action: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
    """
    Sample actions from a list of unnormalized logits and compute associated log-probabilities and entropy.

    IMPORTANT: All logits tensors must have the same batch size and device, but may have different
    numbers of classes (i.e., different shapes in dimension 1).

    Each element in `logits` corresponds to a categorical distribution over one *component* of the full action.
    For example, a 2-component action might involve:
      - logits[0]: action type (e.g., [move, attack, wait]) typ. shape [B,9]
      - logits[1]: action parameter (e.g., [north, south, east, ...]) typ. shape [B,10]

    This function supports either sampling actions from these logits or using externally provided actions.

    Args:
        logits:
            List of unnormalized logits (pre-softmax), one per action component.
            Each tensor has shape [batch_size, num_classes_i], where num_classes_i can vary per component.
        action:
            Optional tensor of shape [A, B, num_components], where A x B = batch_size
            If provided, these actions will be used instead of sampling.

    Returns:
        A tuple of:
            - actions: Tensor of shape [batch_size, num_components], sampled or provided
            - joint_logprob: Tensor of shape [batch_size], sum of log-probabilities across all action components
            - joint_entropy: Tensor of shape [batch_size], sum of entropies across all logits
            - normalized_logits: List of log-softmaxed tensors (same shape as input logits)
    """

    if not isinstance(logits, list):
        logits = [logits]

    # Step 1: Normalize logits into log-probabilities for numerical stability and sampling
    normalized_logits = [F.log_softmax(logit, dim=-1) for logit in logits]  # i.e. log probs
    softmaxed_logits = [logit.exp() for logit in normalized_logits]  # i.e. probs

    num_components = len(logits)
    batch_size = logits[0].shape[0]
    device = logits[0].device

    # Step 2: Determine the actions to evaluate
    if action is None:
        # If no actions are provided, sample from each logit independently

        # Create tensor to hold sampled actions, shape: [num_components, batch_size]
        output_action = torch.empty(num_components, batch_size, dtype=torch.long, device=device)

        # Sample actions from each categorical distribution
        for i, probs in enumerate(softmaxed_logits):
            output_action[i] = torch.multinomial(probs, 1).squeeze(-1)
    else:
        # If actions are provided, reshape them to [num_components, batch_size]
        output_action = action.view(batch_size, -1).T

    # Step 3: Compute log-probabilities for each selected action component
    joint_logprob = torch.zeros(batch_size, device=device)

    # For each action component, extract the log probability of the selected action
    for i in range(num_components):
        # Get the log probability of the selected action for this component
        # output_action[i] has shape [batch_size]
        # normalized_logits[i] has shape [batch_size, num_classes_i]
        # Use a modified version of gather that uses .long() on actions
        component_action = output_action[i].long()

        # Use gather explicitly as it's more reliable in TorchScript
        component_logprob = normalized_logits[i].gather(dim=1, index=component_action.view(-1, 1)).squeeze(1)

        # Sum log-probs across components (equivalent to product of probabilities)
        joint_logprob += component_logprob

    # Step 4: Compute entropy for each categorical component
    joint_entropy = torch.zeros(batch_size, device=device)

    # Sum entropies across all action components
    for i, log_probs in enumerate(normalized_logits):
        probs = softmaxed_logits[i]
        component_entropy = -torch.sum(probs * log_probs, dim=-1)
        joint_entropy += component_entropy

    # Step 5: Return actions in user-facing format [batch_size, num_components]
    return output_action.T, joint_logprob, joint_entropy, normalized_logits
