"""Conservative Model-based Policy Optimization (CMPO).

This implements the CMPO policy improvement component from the Muesli algorithm.
CMPO provides a stable target policy for training by reweighting the prior policy
with clipped advantages.

The CMPO policy is defined as:
    π_cmpo(a|s) ∝ π_prior(a|s) * exp(clip(Â(s,a) / σ²))

where:
- π_prior is the current policy (or target policy)
- Â(s,a) are advantages normalized by their variance σ²
- clip ensures advantages are bounded (typically [-1, 1])

Reference: Hessel et al., 2021 - Muesli: Combining Improvements in Policy Optimization
"""

from typing import Any

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class CMPOConfig(LossConfig):
    """Configuration for CMPO policy loss.

    CMPO adds a KL regularization term that encourages the policy to stay close
    to the CMPO target policy, which is a conservative improvement over the prior.
    """

    # KL regularization coefficient
    kl_coef: float = Field(default=0.1, ge=0.0)

    # Advantage clipping range for CMPO target
    advantage_clip_min: float = Field(default=-1.0)
    advantage_clip_max: float = Field(default=1.0)

    # Whether to use target network for computing prior policy
    use_target_network: bool = Field(default=True)

    # EMA decay for target network
    target_ema_decay: float = Field(default=0.99, ge=0.0, le=1.0)

    # Minimum variance for advantage normalization
    min_advantage_variance: float = Field(default=1e-8, ge=0.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "CMPO":
        """Create CMPO loss instance."""
        return CMPO(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            cfg=self,
        )


class CMPO(Loss):
    """Conservative Model-based Policy Optimization loss.

    Computes the CMPO target policy and adds KL regularization to keep the
    learned policy close to this conservative target.

    The loss is:
        L_cmpo = KL(π_cmpo || π) = Σ π_cmpo(a|s) * log(π_cmpo(a|s) / π(a|s))

    This encourages the policy to match the CMPO target, which is a principled
    way to incorporate advantage information without the instability of raw
    advantage-weighted updates.
    """

    __slots__ = ("target_policy", "advantage_variance_ema")

    def __post_init__(self):
        super().__post_init__()

        # Initialize target network for stable prior policy
        if self.cfg.use_target_network:
            import copy

            self.target_policy = copy.deepcopy(self.policy)
            for param in self.target_policy.parameters():
                param.requires_grad = False
        else:
            self.target_policy = None

        # Track advantage variance with EMA for normalization
        self.advantage_variance_ema = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.register_state_attr("advantage_variance_ema")

    def update_target_network(self) -> None:
        """Update target network with EMA."""
        if self.target_policy is None:
            return

        with torch.no_grad():
            target_state = self.target_policy.state_dict()
            online_state = self.policy.state_dict()

            for name, online_param in online_state.items():
                if name in target_state:
                    target_param = target_state[name]
                    if target_param.shape == online_param.shape:
                        target_state[name] = (
                            self.cfg.target_ema_decay * target_param + (1 - self.cfg.target_ema_decay) * online_param
                        )
                    else:
                        # Shape mismatch (e.g., lazy init resize) - copy directly
                        target_state[name] = online_param.clone()
                else:
                    # New parameter - add it
                    target_state[name] = online_param.clone()

            self.target_policy.load_state_dict(target_state)

    def compute_cmpo_policy(
        self,
        logits: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """Compute CMPO target policy: π_cmpo(a|s) ∝ π_prior(a|s) * exp(clip(Â(s,a) / σ²))

        Args:
            logits: Policy logits from prior/target network (B, T, A)
            advantages: Advantages for each action (B, T, A) or (B, T)

        Returns:
            CMPO target policy probabilities (B, T, A)
        """
        # Get prior policy probabilities
        pi_prior = F.softmax(logits, dim=-1)

        # Update advantage variance EMA
        with torch.no_grad():
            current_var = advantages.var().clamp(min=self.cfg.min_advantage_variance)
            self.advantage_variance_ema = 0.99 * self.advantage_variance_ema + 0.01 * current_var

        # Normalize advantages by variance
        adv_normalized = advantages / (self.advantage_variance_ema + self.cfg.min_advantage_variance)

        # Clip advantages
        adv_clipped = torch.clamp(
            adv_normalized,
            self.cfg.advantage_clip_min,
            self.cfg.advantage_clip_max,
        )

        # Expand advantages to match action dimension if needed
        if adv_clipped.ndim < pi_prior.ndim:
            adv_clipped = adv_clipped.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        # CMPO: reweight prior by exponentiated clipped advantage
        # π_cmpo(a|s) ∝ π_prior(a|s) * exp(clip(Â(s,a) / σ²))
        pi_cmpo = pi_prior * torch.exp(adv_clipped)

        # Normalize to valid probability distribution
        pi_cmpo = pi_cmpo / (pi_cmpo.sum(dim=-1, keepdim=True) + 1e-8)

        return pi_cmpo

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute CMPO loss.

        Args:
            shared_loss_data: Shared data including sampled minibatch and policy outputs
            context: Training context
            mb_idx: Minibatch index

        Returns:
            Tuple of (loss, updated_shared_data, stop_flag)
        """
        # Update target network on first minibatch
        if mb_idx == 0:
            self.update_target_network()

        # Get data from shared loss data
        minibatch = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data["policy_td"]

        # Get advantages
        advantages = minibatch["advantages"]  # (B, T)

        # Get current policy logits
        current_logits = policy_td.get("logits")

        # Get prior policy logits (from target network or current policy)
        if self.target_policy is not None:
            with torch.no_grad():
                # Need to run target policy on the minibatch
                # For simplicity, use current logits as approximation
                # In a full implementation, you'd re-forward through target_policy
                prior_logits = current_logits.detach()
        else:
            prior_logits = current_logits.detach()

        # Compute CMPO target policy
        with torch.no_grad():
            pi_cmpo = self.compute_cmpo_policy(prior_logits, advantages)

        # Store CMPO policy in shared data for use by other losses (e.g., Muesli model loss)
        shared_loss_data["cmpo_policies"] = pi_cmpo

        # KL divergence: KL(π_cmpo || π_current) = Σ π_cmpo * log(π_cmpo / π_current)
        # This is equivalent to cross-entropy loss
        log_pi_current = F.log_softmax(current_logits, dim=-1)
        kl_loss = -(pi_cmpo * log_pi_current).sum(dim=-1).mean()

        # Total CMPO loss
        cmpo_loss = self.cfg.kl_coef * kl_loss

        # Track metrics
        self.loss_tracker["cmpo_kl_loss"].append(float(kl_loss.item()))
        self.loss_tracker["cmpo_loss"].append(float(cmpo_loss.item()))

        # Track advantage statistics
        with torch.no_grad():
            self.loss_tracker["cmpo_adv_mean"].append(float(advantages.mean().item()))
            self.loss_tracker["cmpo_adv_std"].append(float(advantages.std().item()))
            self.loss_tracker["cmpo_adv_variance_ema"].append(float(self.advantage_variance_ema.item()))

        return cmpo_loss, shared_loss_data, False
