from typing import Optional, Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from unnormalized logits and compute associated log-probabilities and entropy.
    `logits` corresponds to a categorical distribution over the full action.
    This function supports either sampling actions from these logits or using externally provided actions.

    Args:
        logits:
            unnormalized logits (pre-softmax)
        action:
            Optional tensor of shape [batch_size, 1]
            If provided, these actions will be used instead of sampling.

    Returns:
        A tuple of:
        - actions: Tensor of shape [batch_size, 1], sampled or provided
        - joint_logprob: Tensor of shape [batch_size], log-probabilities of selected actions
        - joint_entropy: Tensor of shape [batch_size], entropy of the distribution
        - normalized_logits: Log-softmaxed logits (same shape as input logits)
    """
    # Step 1: Normalize logits into log-probabilities for numerical stability and sampling
    normalized_logits = F.log_softmax(logits, dim=-1)  # i.e. log probs
    softmaxed_logits = normalized_logits.exp()  # i.e. probs
    batch_size = logits.shape[0]

    # Step 2: Determine the actions to evaluate
    if action is None:
        # Sample actions from the categorical distribution
        output_action = torch.multinomial(softmaxed_logits, 1)
    else:
        # Use provided actions, ensure correct shape
        output_action = action.view(batch_size, 1)

    # Step 3: Compute log-probabilities for the selected actions
    # Gather the log probability of each selected action
    indices = output_action.long()
    joint_logprob = normalized_logits.gather(dim=1, index=indices).squeeze(1)

    # Step 4: Compute entropy directly without a loop
    # Entropy formula: -sum(p * log(p))
    joint_entropy = -torch.sum(softmaxed_logits * normalized_logits, dim=-1)

    return output_action, joint_logprob, joint_entropy, normalized_logits
