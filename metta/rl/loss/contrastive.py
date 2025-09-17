# metta/rl/loss/contrastive.py
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.metta_agent import PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.trainer_state import TrainerState


class ContrastiveLoss(BaseLoss):
    """Contrastive loss for representation learning."""

    __slots__ = (
        "temperature",
        "contrastive_coef",
        "embedding_dim",
        "projection_head",
        "_projection_head_input_dim",
        "_value_projection",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, checkpoint_manager, instance_name, loss_config)

        self.temperature = self.loss_cfg.temperature
        self.contrastive_coef = self.loss_cfg.contrastive_coef
        self.embedding_dim = self.loss_cfg.embedding_dim

        # Add projection head if needed
        if self.loss_cfg.use_projection_head:
            # We'll determine input_dim dynamically during first forward pass
            # This avoids hardcoded assumptions about encoder output dimensions
            self.projection_head = None  # Will be created in first forward pass
            self._projection_head_input_dim = None
            print("Contrastive loss projection head will be created dynamically based on actual embedding dimensions")
        else:
            self.projection_head = None

    def get_experience_spec(self) -> Composite:
        """Define additional data needed for contrastive learning."""
        return Composite(
            # Add any additional data needed for contrastive learning
            # e.g., positive/negative pairs, augmentations, etc.
        )

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Compute contrastive loss."""
        policy_td = shared_loss_data["policy_td"]
        minibatch = shared_loss_data["sampled_mb"]

        # Get embeddings from policy
        embeddings = self._get_embeddings(policy_td)

        # Create and apply projection head if needed
        if self.loss_cfg.use_projection_head:
            if self.projection_head is None:
                # Create projection head dynamically based on actual embedding dimensions
                actual_input_dim = embeddings.shape[-1]
                self._projection_head_input_dim = actual_input_dim
                self.projection_head = torch.nn.Linear(actual_input_dim, self.embedding_dim).to(self.device)
                print(f"Created contrastive loss projection head: {actual_input_dim} -> {self.embedding_dim}")

            embeddings = self.projection_head(embeddings)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(embeddings, minibatch)

        # Track metrics
        self.loss_tracker["contrastive_loss"].append(float(contrastive_loss.item()))

        return contrastive_loss, shared_loss_data

    def _get_embeddings(self, policy_td: TensorDict) -> Tensor:
        """Extract embeddings from policy output."""
        # Try different possible embedding sources in order of preference
        if "encoder_output" in policy_td:
            return policy_td["encoder_output"]
        elif "hidden_state" in policy_td:
            return policy_td["hidden_state"]
        elif "features" in policy_td:
            return policy_td["features"]
        else:
            # Fallback: use value as embeddings but warn about suboptimal choice
            # This should only happen if policy doesn't provide proper feature representations
            print(
                "WARNING: Contrastive loss using value tensor as embeddings - "
                "this is suboptimal for representation learning"
            )
            value = policy_td["value"].squeeze(-1)  # Remove last dimension if it's 1

            if value.dim() == 1:
                # Don't expand identical values - instead create a learnable linear projection
                # from the 1D value to embedding_dim with proper initialization
                if not hasattr(self, "_value_projection"):
                    self._value_projection = torch.nn.Linear(1, self.embedding_dim).to(self.device)
                    # Initialize with small random weights to break symmetry
                    torch.nn.init.xavier_uniform_(self._value_projection.weight)
                    print(f"Created value->embedding projection: 1 -> {self.embedding_dim}")

                # Project 1D value to embedding_dim with learned transformation
                value = value.unsqueeze(-1)  # [N] -> [N, 1]
                value = self._value_projection(value)  # [N, 1] -> [N, embedding_dim]

            return value

    def _compute_contrastive_loss(self, embeddings: Tensor, minibatch: TensorDict) -> Tensor:
        """Compute the actual contrastive loss using proper positive/negative sampling."""
        batch_size = embeddings.shape[0]

        # Reshape to (B*T, D) if needed
        if embeddings.dim() == 3:  # (B, T, D)
            B, T, D = embeddings.shape
            embeddings = embeddings.view(B * T, D)
            batch_size = B * T

        if batch_size < 4:  # Need at least 4 samples for proper contrastive learning
            return torch.tensor(0.0, device=self.device)

        # Create proper positive and negative pairs
        # Positive pairs: consecutive timesteps (temporal continuity)
        # Negative pairs: randomly shuffled embeddings (different contexts)

        anchor_embeddings = embeddings[:-1]  # All except last
        pos_embeddings = embeddings[1:]  # All except first (consecutive timesteps)

        # Create negative embeddings by randomly shuffling indices
        # This ensures negatives are from different contexts, not identical tensors
        neg_indices = torch.randperm(len(pos_embeddings), device=self.device)
        # Ensure negatives are actually different from positives
        # If by chance neg_indices[i] == i, shift by 1
        same_indices = neg_indices == torch.arange(len(neg_indices), device=self.device)
        neg_indices[same_indices] = (neg_indices[same_indices] + 1) % len(neg_indices)
        neg_embeddings = pos_embeddings[neg_indices]

        # Compute similarities
        pos_sim = F.cosine_similarity(anchor_embeddings, pos_embeddings, dim=-1)
        neg_sim = F.cosine_similarity(anchor_embeddings, neg_embeddings, dim=-1)

        # InfoNCE-style contrastive loss
        # Maximize positive similarity, minimize negative similarity
        pos_logits = pos_sim / self.temperature
        neg_logits = neg_sim / self.temperature

        # Contrastive loss: log(exp(pos) / (exp(pos) + exp(neg)))
        # This is equivalent to: -log(sigmoid(pos - neg))
        contrastive_loss = -torch.log(torch.sigmoid(pos_logits - neg_logits)).mean()

        return contrastive_loss * self.contrastive_coef
