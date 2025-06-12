from typing import Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_actions_with_kl(action_logits: Tensor, target_probs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference, computing KL divergence from target distribution.

    Args:
        action_logits: Raw logits from policy network of shape [batch_size, num_actions].
                       These are unnormalized log-probabilities over the action space.
        target_probs: Target probability distribution over actions, shape [num_actions].
                      Should sum to 1.0.

    Returns:
        actions: Sampled action indices of shape [batch_size]. Each element is an
                 integer in [0, num_actions) representing the sampled action.

        log_probs: Log-probabilities of the sampled actions, shape [batch_size].

        kl_divergence: KL divergence from target distribution at each state, shape [batch_size].
                       KL(π||p_target) = Σ π(a|s) log(π(a|s)/p_target(a))

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

    # Sample actions from categorical distribution
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute KL divergence: KL(π||p_target) = Σ π(a|s) log(π(a|s)/p_target(a))
    # This expands to: Σ π(a|s) [log π(a|s) - log p_target(a)]
    target_log_probs = torch.log(target_probs + 1e-10)  # Add small epsilon for numerical stability
    kl_divergence = torch.sum(action_probs * (action_log_probs - target_log_probs), dim=-1)  # [batch_size]

    return actions, log_probs, kl_divergence, action_log_probs


@torch.jit.script
def evaluate_actions_with_kl(
    action_logits: Tensor, actions: Tensor, target_probs: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate provided actions against logits during training, computing KL divergence from target.

    Args:
        action_logits: Current policy logits of shape [batch_size, num_actions].
                       These may differ from the logits that originally generated
                       the actions due to policy updates.

        actions: Previously taken action indices of shape [batch_size].
                 Each element must be a valid action index in [0, num_actions).

        target_probs: Target probability distribution over actions, shape [num_actions].
                      Should sum to 1.0.

    Returns:
        log_probs: Log-probabilities of the given actions under current policy,
                   shape [batch_size]. Used for importance sampling: π_new(a|s)/π_old(a|s).

        kl_divergence: KL divergence from target distribution at each state, shape [batch_size].

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute KL divergence
    target_log_probs = torch.log(target_probs + 1e-10)
    kl_divergence = torch.sum(action_probs * (action_log_probs - target_log_probs), dim=-1)  # [batch_size]

    return log_probs, kl_divergence, action_log_probs


# Keep the original functions for backward compatibility
@torch.jit.script
def sample_actions(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference.

    Args:
        action_logits: Raw logits from policy network of shape [batch_size, num_actions].
                       These are unnormalized log-probabilities over the action space.

    Returns:
        actions: Sampled action indices of shape [batch_size]. Each element is an
                 integer in [0, num_actions) representing the sampled action.

        log_probs: Log-probabilities of the sampled actions, shape [batch_size].

        entropy: Policy entropy at each state, shape [batch_size].

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

    # Sample actions from categorical distribution (replacement=True is implicit when num_samples=1)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -Σπ(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]

    return actions, log_probs, entropy, action_log_probs


@torch.jit.script
def evaluate_actions(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate provided actions against logits during training.

    Args:
        action_logits: Current policy logits of shape [batch_size, num_actions].
                       These may differ from the logits that originally generated
                       the actions due to policy updates.

        actions: Previously taken action indices of shape [batch_size].
                 Each element must be a valid action index in [0, num_actions).

    Returns:
        log_probs: Log-probabilities of the given actions under current policy,
                   shape [batch_size]. Used for importance sampling: π_new(a|s)/π_old(a|s).

        entropy: Current policy entropy at each state, shape [batch_size].

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -Σπ(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]

    return log_probs, entropy, action_log_probs
