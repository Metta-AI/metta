"""VAPOR: Variational Policy Optimization for RL as Inference

This module implements the VAPOR algorithm from "Probabilistic Inference in
Reinforcement Learning Done Right" (Tarbouriech et al., 2023).

VAPOR treats RL as variational inference over the posterior probability of
state-action optimality, resulting in principled exploration and often improved
stability compared to standard policy gradient methods.
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class VAPORLoss:
    """VAPOR (Variational Policy Optimization) loss implementation.

    VAPOR reformulates RL as variational inference over state-action optimality.
    The key insight is to maintain a variational approximation to the posterior
    probability that each state-action pair is optimal, then use this for both
    policy updates and exploration.
    """

    def __init__(
        self,
        beta: float = 1.0,
        beta_schedule: str = "constant",
        min_beta: float = 0.1,
        exploration_bonus: float = 0.1,
        use_importance_weighting: bool = True,
    ):
        """Initialize VAPOR loss.

        Args:
            beta: KL regularization coefficient (temperature parameter)
            beta_schedule: How to anneal beta ("constant", "linear", "exponential")
            min_beta: Minimum beta value during annealing
            exploration_bonus: Bonus for state-action uncertainty (exploration)
            use_importance_weighting: Whether to use importance weighting for off-policy correction
        """
        self.beta = beta
        self.initial_beta = beta
        self.beta_schedule = beta_schedule
        self.min_beta = min_beta
        self.exploration_bonus = exploration_bonus
        self.use_importance_weighting = use_importance_weighting

        # Track training progress for beta scheduling
        self.step_count = 0

    def update_beta(self, progress: float):
        """Update beta according to schedule.

        Args:
            progress: Training progress in [0, 1]
        """
        if self.beta_schedule == "constant":
            return
        elif self.beta_schedule == "linear":
            self.beta = max(self.min_beta, self.initial_beta * (1.0 - progress))
        elif self.beta_schedule == "exponential":
            self.beta = max(self.min_beta, self.initial_beta * (0.5**progress))
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def compute_posterior_approximation(
        self, logits: Tensor, advantages: Tensor, old_logprobs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute variational approximation to state-action optimality posterior.

        Args:
            logits: Current policy logits [B, A]
            advantages: Advantage estimates [B]
            old_logprobs: Old policy log probabilities [B]

        Returns:
            Tuple of (approximate_posterior, uncertainty)
        """
        batch_size = logits.shape[0]

        # Convert advantages to "optimality signals"
        # Higher advantages suggest higher probability of optimality
        optimality_scores = advantages.detach()

        # Normalize to get approximate posterior over optimality
        # Using softmax with temperature (beta) for exploration control
        posterior_logits = optimality_scores / self.beta
        approximate_posterior = F.softmax(posterior_logits, dim=0)

        # Compute uncertainty as entropy of posterior
        posterior_entropy = -(approximate_posterior * torch.log(approximate_posterior + 1e-8)).sum()
        uncertainty = posterior_entropy / torch.log(torch.tensor(batch_size, dtype=torch.float32))

        return approximate_posterior, uncertainty

    def compute_variational_policy_loss(
        self, new_logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, logits: Tensor, importance_ratio: Tensor
    ) -> Tuple[Tensor, dict]:
        """Compute VAPOR policy loss using variational inference.

        Args:
            new_logprobs: New policy log probabilities [B]
            old_logprobs: Old policy log probabilities [B]
            advantages: Advantage estimates [B]
            logits: Policy logits [B, A]
            importance_ratio: Importance sampling ratios [B]

        Returns:
            Tuple of (policy_loss, info_dict)
        """
        # Compute posterior approximation
        posterior, uncertainty = self.compute_posterior_approximation(logits, advantages, old_logprobs)

        # VAPOR policy gradient with variational approximation
        # The key insight: weight policy gradients by posterior optimality probability
        if self.use_importance_weighting:
            weighted_advantages = importance_ratio * advantages * posterior[torch.arange(len(advantages))]
        else:
            weighted_advantages = advantages * posterior[torch.arange(len(advantages))]

        # Policy loss: negative expected return under variational posterior
        policy_loss = -(new_logprobs * weighted_advantages).mean()

        # KL regularization term (prevents policy from changing too quickly)
        kl_div = new_logprobs - old_logprobs
        kl_penalty = self.beta * kl_div.mean()

        # Exploration bonus based on posterior uncertainty
        exploration_bonus = -self.exploration_bonus * uncertainty

        # Total VAPOR loss
        total_loss = policy_loss + kl_penalty + exploration_bonus

        # Information for logging
        info = {
            "vapor_policy_loss": policy_loss.item(),
            "vapor_kl_penalty": kl_penalty.item(),
            "vapor_exploration_bonus": exploration_bonus.item(),
            "vapor_posterior_entropy": uncertainty.item(),
            "vapor_beta": self.beta,
        }

        return total_loss, info

    def compute_variational_value_loss(
        self, values: Tensor, returns: Tensor, posterior_weights: Tensor = None
    ) -> Tensor:
        """Compute value loss with optional posterior weighting.

        Args:
            values: Value predictions [B]
            returns: Target returns [B]
            posterior_weights: Optional posterior optimality weights [B]

        Returns:
            Value loss tensor
        """
        value_errors = (values - returns) ** 2

        if posterior_weights is not None:
            # Weight value learning by posterior optimality probability
            weighted_errors = posterior_weights * value_errors
            return weighted_errors.mean()
        else:
            return value_errors.mean()


def compute_vapor_losses(
    minibatch: dict,
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    logits: Tensor,
    vapor_config: dict,
    device: torch.device,
    training_progress: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Compute VAPOR losses for policy and value functions.

    Args:
        minibatch: Batch of experience data
        new_logprobs: New policy log probabilities
        entropy: Policy entropy
        newvalue: New value predictions
        importance_sampling_ratio: Importance sampling ratios
        adv: Advantage estimates
        logits: Policy logits
        vapor_config: VAPOR configuration dict
        device: Torch device
        training_progress: Training progress for beta scheduling

    Returns:
        Tuple of (policy_loss, value_loss, entropy_loss, vapor_info)
    """
    # Initialize VAPOR loss computer
    vapor_loss = VAPORLoss(
        beta=vapor_config.beta,
        beta_schedule=vapor_config.beta_schedule,
        min_beta=vapor_config.min_beta,
        exploration_bonus=vapor_config.exploration_bonus,
        use_importance_weighting=vapor_config.use_importance_weighting,
    )

    # Update beta based on training progress
    vapor_loss.update_beta(training_progress)

    # Compute VAPOR policy loss
    policy_loss, vapor_info = vapor_loss.compute_variational_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=minibatch["logprobs"],
        advantages=adv,
        logits=logits,
        importance_ratio=importance_sampling_ratio,
    )

    # Compute posterior for value loss weighting
    posterior, _ = vapor_loss.compute_posterior_approximation(logits, adv, minibatch["logprobs"])

    # VAPOR value loss with posterior weighting
    value_loss = vapor_loss.compute_variational_value_loss(
        values=newvalue.view(minibatch["returns"].shape),
        returns=minibatch["returns"],
        posterior_weights=posterior[torch.arange(len(minibatch["returns"]))],
    )

    # Entropy loss (standard)
    entropy_loss = entropy.mean()

    return policy_loss, value_loss, entropy_loss, vapor_info
