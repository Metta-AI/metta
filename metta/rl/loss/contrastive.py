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
            # Check if encoder component exists before accessing it
            if "encoder" in self.policy.policy.components:
                input_dim = self.policy.policy.components["encoder"].output_dim
                print(f"Using encoder output_dim: {input_dim}")
            else:
                # Fallback: use a reasonable default based on policy architecture
                input_dim = 256  # Reasonable default for policy representations
                print(f"Encoder not found, using fallback input_dim: {input_dim}")
            self.projection_head = torch.nn.Linear(input_dim, self.embedding_dim).to(device)
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

        # Apply projection head if needed
        if self.projection_head is not None:
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
            # Fallback: use value as embeddings (expand to reasonable dimension)
            value = policy_td["value"].squeeze(-1)  # Remove last dimension if it's 1
            if value.dim() == 1:
                # If 1D, repeat to create a reasonable embedding dimension
                value = value.unsqueeze(-1).expand(-1, self.embedding_dim)
            return value

    def _compute_contrastive_loss(self, embeddings: Tensor, minibatch: TensorDict) -> Tensor:
        """Compute the actual contrastive loss."""
        batch_size = embeddings.shape[0]

        # For simplicity, using temporal contrastive learning
        # Positive pairs: consecutive timesteps
        # Negative pairs: non-consecutive timesteps

        # Reshape to (B*T, D) if needed
        if embeddings.dim() == 3:  # (B, T, D)
            B, T, D = embeddings.shape
            embeddings = embeddings.view(B * T, D)
            batch_size = B * T

        # Create positive pairs (consecutive timesteps)
        if batch_size > 1:
            # For each embedding, the next one is positive
            pos_embeddings = embeddings[1:]  # Skip first
            neg_embeddings = embeddings[:-1]  # Skip last

            # Compute similarities
            pos_sim = F.cosine_similarity(embeddings[:-1], pos_embeddings, dim=-1)
            neg_sim = F.cosine_similarity(embeddings[:-1], neg_embeddings, dim=-1)

            # Contrastive loss: maximize positive similarity, minimize negative
            pos_loss = -torch.log(torch.sigmoid(pos_sim / self.temperature)).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_sim / self.temperature)).mean()

            contrastive_loss = (pos_loss + neg_loss) * self.contrastive_coef
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)

        return contrastive_loss
