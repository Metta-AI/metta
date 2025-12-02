"""Muesli Model Loss (L_m) - Policy Prediction Component.

This implements the policy prediction component of the Muesli algorithm (Hessel et al., 2021).
The key innovation of Muesli is training the model to predict future POLICIES in addition to
values and rewards.

This loss complements the existing Dynamics loss (dynamics.py) by adding:
- Policy prediction loss: trains model to predict policy at future timesteps
- CMPO target integration: uses conservative target policies for training

Note: For value/reward prediction, use the existing Dynamics loss. This loss focuses on
the policy prediction aspect that makes Muesli unique.

Reference: https://arxiv.org/abs/2104.06159
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


class MuesliModelConfig(LossConfig):
    """Configuration for Muesli policy prediction loss.

    This loss trains the model to predict future policies, which makes the learned
    representations "policy-aware". This complements value/reward prediction from
    the Dynamics loss.
    """

    # Loss coefficient for policy prediction
    # Reduced from 1.0 to 0.001 because cross-entropy loss is ~5.0 (much larger than other losses)
    policy_pred_coef: float = Field(default=0.001, ge=0.0)

    # Number of future timesteps to predict (K in paper)
    # Setting this to 1 means predict policy at current timestep only
    # Setting to 5 means predict policies 1-5 steps ahead
    policy_horizon: int = Field(default=0, ge=1, le=20)

    @property
    def enabled(self) -> bool:
        return self.policy_horizon > 0

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "MuesliModel":
        """Create Muesli model loss instance."""
        return MuesliModel(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            cfg=self,
        )


class MuesliModel(Loss):
    """Muesli policy prediction loss.

    The key innovation of Muesli: train the model to predict future policies.
    This makes the learned representations "policy-aware" and useful for planning.

    Works with standard policy outputs:
    - "logits": policy logits (B, T, num_actions)

    Uses CMPO target policies if available in shared_loss_data, otherwise uses
    current policy as target.
    """

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute Muesli policy prediction loss.

        Trains the model to predict policies at future timesteps, making the
        learned representations "policy-aware".

        Args:
            shared_loss_data: Shared data including sampled minibatch and policy outputs
            context: Training context
            mb_idx: Minibatch index

        Returns:
            Tuple of (loss, updated_shared_data, stop_flag)
        """
        policy_td = shared_loss_data["policy_td"]
        assert isinstance(self.cfg, MuesliModelConfig)
        assert isinstance(policy_td, TensorDict)
        B, T = policy_td.batch_size

        # Get policy logits
        logits = policy_td.get("logits")
        if logits is None:
            raise RuntimeError(
                f"Muesli model loss requires 'logits' in policy output.\n"
                f"Available keys: {list(policy_td.keys())}\n"
                f"Policy shape: {policy_td.batch_size}"
            )

        with torch.no_grad():
            target_policies = F.softmax(logits.detach(), dim=-1)

        # Compute policy prediction loss over K timesteps
        total_policy_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        unrolled_logits = shared_loss_data.get("muesli_unrolled_logits")
        assert unrolled_logits is not None, "muesli_unrolled_logits missing from shared_loss_data"

        # Restore shape: (B, T, K, A) -> (K, B, T, A)
        unrolled_logits = unrolled_logits.permute(2, 0, 1, 3)

        # Use unrolled predictions from Dynamics loss
        K_unrolled = unrolled_logits.shape[0]

        for k in range(K_unrolled):
            logits_k = unrolled_logits[k]  # (B, T_eff, A)
            T_eff = logits_k.shape[1]

            # Targets at t+k+1
            # Ensure alignment
            if k + 1 + T_eff <= target_policies.shape[1]:
                target_k = target_policies[:, k + 1 : k + 1 + T_eff]
            else:
                target_k = target_policies[:, k + 1 :]
                logits_k = logits_k[:, : target_k.shape[1]]

            logits_flat = logits_k.reshape(-1, logits_k.shape[-1])
            target_flat = target_k.reshape(-1, target_k.shape[-1])

            log_probs = F.log_softmax(logits_flat, dim=-1)
            policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()

            total_policy_loss += policy_loss

        avg_policy_loss = total_policy_loss / K_unrolled

        weighted_loss = avg_policy_loss * self.cfg.policy_pred_coef

        # Track metrics
        assert self.loss_tracker is not None
        self.loss_tracker["muesli_policy_loss"].append(float(weighted_loss.item()))

        return weighted_loss, shared_loss_data, False
