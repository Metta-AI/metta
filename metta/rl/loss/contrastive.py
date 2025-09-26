# metta/rl/loss/contrastive.py
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment


class ContrastiveLoss(Loss):
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
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_config)

        self.temperature = self.loss_cfg.temperature
        self.contrastive_coef = self.loss_cfg.contrastive_coef
        self.embedding_dim = self.loss_cfg.embedding_dim

        # Add projection head if needed
        if self.loss_cfg.use_projection_head:
            # We'll determine input_dim dynamically during first forward pass
            # This avoids hardcoded assumptions about encoder output dimensions
            self.projection_head = None  # Will be created in first forward pass
            self._projection_head_input_dim = None
        else:
            self.projection_head = None

    def get_experience_spec(self) -> Composite:
        """Define additional data needed for contrastive learning."""
        return Composite(
            # Add any additional data needed for contrastive learning
            # e.g., positive/negative pairs, augmentations, etc.
        )

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
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

            embeddings = self.projection_head(embeddings)

        # Compute InfoNCE contrastive loss (normalization handled internally)
        contrastive_loss, metrics = self._compute_contrastive_loss(embeddings, minibatch)

        # Track metrics
        self.loss_tracker["contrastive_loss"].append(float(contrastive_loss.item()))

        # Track additional metrics for wandb logging
        for key, value in metrics.items():
            if key not in self.loss_tracker:
                self.loss_tracker[key] = []
            self.loss_tracker[key].append(value)

        return contrastive_loss, shared_loss_data, False

    def _get_embeddings(self, policy_td: TensorDict) -> Tensor:
        """Extract embeddings from policy output."""
        # Try different possible embedding sources in order of preference
        if "encoder_output" in policy_td:
            return policy_td["encoder_output"]
        elif "encoded_obs" in policy_td:
            return policy_td["encoded_obs"]  # Try encoded observations
        elif "core" in policy_td:
            return policy_td["core"]  # Try core hidden state
        elif "hidden_state" in policy_td:
            return policy_td["hidden_state"]
        elif "features" in policy_td:
            return policy_td["features"]
        else:
            # Fallback: use value as embeddings but warn about suboptimal choice
            # This should only happen if policy doesn't provide proper feature representations
            value = policy_td["values"].squeeze(-1)  # Remove last dimension if it's 1

            if value.dim() == 1:
                # Don't expand identical values - instead create a learnable linear projection
                # from the 1D value to embedding_dim with proper initialization
                if not hasattr(self, "_value_projection"):
                    self._value_projection = torch.nn.Linear(1, self.embedding_dim).to(self.device)
                    # Initialize with small random weights to break symmetry
                    torch.nn.init.xavier_uniform_(self._value_projection.weight)

                # Project 1D value to embedding_dim with learned transformation
                value = value.unsqueeze(-1)  # [N] -> [N, 1]
                value = self._value_projection(value)  # [N, 1] -> [N, embedding_dim]

            return value

    def _compute_contrastive_loss(self, embeddings: Tensor, minibatch: TensorDict) -> tuple[Tensor, dict]:
        """Compute InfoNCE contrastive loss following standard implementations."""
        batch_size = embeddings.shape[0]

        # Reshape to (B*T, D) if needed
        if embeddings.dim() == 3:  # (B, T, D)
            B, T, D = embeddings.shape
            embeddings = embeddings.view(B * T, D)
            batch_size = B * T

        if batch_size < 4:  # Need minimum samples for contrastive learning
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "num_pairs": 0,
            }

        # L2 normalize embeddings to unit vectors (critical for InfoNCE)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Create temporal positive pairs (consecutive timesteps for RL continuity)
        anchor_embeddings = embeddings[:-1]  # [N-1, D]
        positive_embeddings = embeddings[1:]  # [N-1, D] consecutive timesteps

        # InfoNCE: Use in-batch negatives for efficiency
        # Each anchor's negatives are all other positives in the batch
        num_pairs = len(anchor_embeddings)

        if num_pairs < 2:
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "num_pairs": 0,
            }

        # Compute positive similarities: [N-1]
        positive_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=-1)

        # Compute negative similarities: [N-1, N-1]
        # Each anchor compared with ALL other positives as negatives
        negative_sim = torch.matmul(anchor_embeddings, positive_embeddings.T)

        # Remove self-similarity on diagonal (anchor vs its own positive)
        mask = torch.eye(num_pairs, device=self.device).bool()
        negative_sim = negative_sim.masked_fill(mask, float("-inf"))

        # InfoNCE loss computation
        # Logits: [positive_sim, negative_sim_1, negative_sim_2, ...]
        logits = (
            torch.cat(
                [
                    positive_sim.unsqueeze(1),  # [N-1, 1]
                    negative_sim,  # [N-1, N-1]
                ],
                dim=1,
            )
            / self.temperature
        )  # Apply temperature scaling

        # Labels: positive is always index 0
        labels = torch.zeros(num_pairs, dtype=torch.long, device=self.device)

        # InfoNCE = CrossEntropy(logits, labels) where positive is at index 0
        infonce_loss = F.cross_entropy(logits, labels, reduction="mean")

        # Compute metrics for logging
        positive_sim_mean = positive_sim.mean().item()
        # For negative similarities, exclude the masked values (-inf)
        valid_negative_sim = negative_sim[~mask]
        negative_sim_mean = valid_negative_sim.mean().item()

        metrics = {
            "positive_sim_mean": positive_sim_mean,
            "negative_sim_mean": negative_sim_mean,
            "num_pairs": num_pairs,
            "positive_sim_std": positive_sim.std().item(),
            "negative_sim_std": valid_negative_sim.std().item(),
        }

        return infonce_loss * self.contrastive_coef, metrics
