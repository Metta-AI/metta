# metta/rl/loss/contrastive.py
from typing import Any

import torch
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
        "embedding_key",
        "projection_head",
        "_projection_head_input_dim",
        "_value_projection",
        "discount",
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
        self.embedding_key = getattr(self.loss_cfg, "embedding_key", None)
        self.discount = self.loss_cfg.discount

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

        batch_shape = minibatch.batch_size
        if len(batch_shape) != 2:
            raise ValueError("Contrastive loss expects minibatch with 2D batch size (segments, horizon).")

        segments, horizon = batch_shape

        if embeddings.dim() == 3:
            if embeddings.shape[0] != segments or embeddings.shape[1] != horizon:
                raise ValueError(
                    "Embeddings shape must align with minibatch dimensions. "
                    f"Expected ({segments}, {horizon}, *), received {tuple(embeddings.shape)}."
                )
        elif embeddings.dim() == 2 and embeddings.shape[0] == segments * horizon:
            embeddings = embeddings.reshape(segments, horizon, -1)
        else:
            raise ValueError(
                "Embeddings must have shape [segments, horizon, dim] or [segments * horizon, dim], "
                f"received {tuple(embeddings.shape)} for batch size {(segments, horizon)}."
            )

        # Create and apply projection head if needed
        if self.loss_cfg.use_projection_head:
            if self.projection_head is None:
                # Create projection head dynamically based on actual embedding dimensions
                actual_input_dim = embeddings.shape[-1]
                self._projection_head_input_dim = actual_input_dim
                self.projection_head = torch.nn.Linear(actual_input_dim, self.embedding_dim).to(self.device)

            embeddings = self.projection_head(embeddings)

        # Compute InfoNCE contrastive loss
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
        if self.embedding_key is not None:
            if self.embedding_key not in policy_td.keys(True):
                raise KeyError(f"Contrastive loss expects '{self.embedding_key}' in policy_td")
            return policy_td[self.embedding_key]

        # Try different possible embedding sources in order of preference
        for candidate in ("encoder_output", "encoded_obs", "core", "hidden_state", "features"):
            if candidate in policy_td.keys(True):
                return policy_td[candidate]

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
        """Compute InfoNCE contrastive loss with geometric future positives and shuffled negatives."""

        batch_shape = minibatch.batch_size
        if len(batch_shape) != 2:
            raise ValueError("Contrastive loss expects minibatch with 2D batch size (segments, horizon).")

        segments, horizon = batch_shape

        embedding_dim = embeddings.shape[-1]
        if embedding_dim == 0:
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "positive_sim_std": 0.0,
                "negative_sim_std": 0.0,
                "num_pairs": 0,
                "delta_mean": 0.0,
            }

        dones = minibatch.get("dones")
        if dones is None:
            raise KeyError("Contrastive loss requires 'dones' in minibatch for trajectory boundaries.")
        dones = dones.squeeze(-1) if dones.dim() == 3 else dones
        done_mask = dones.to(dtype=torch.bool)

        truncateds = minibatch.get("truncateds")
        if truncateds is not None:
            truncateds = truncateds.squeeze(-1) if truncateds.dim() == 3 else truncateds
            done_mask = torch.logical_or(done_mask, truncateds.to(dtype=torch.bool))

        done_mask_cpu = done_mask.detach().to("cpu")

        prob = max(1.0 - float(self.discount), 1e-8)
        geom_dist = torch.distributions.Geometric(probs=torch.tensor(prob, device=self.device, dtype=embeddings.dtype))

        batch_indices: list[int] = []
        anchor_steps: list[int] = []
        positive_steps: list[int] = []
        sampled_deltas: list[float] = []

        for batch_idx in range(segments):
            done_row = done_mask_cpu[batch_idx].view(-1)
            episode_bounds: list[tuple[int, int]] = []
            start = 0
            for step, done in enumerate(done_row.tolist()):
                if done:
                    episode_bounds.append((start, step))
                    start = step + 1
            if start < horizon:
                episode_bounds.append((start, horizon - 1))

            candidate_anchors: list[tuple[int, int]] = []
            for episode_start, episode_end in episode_bounds:
                if episode_end - episode_start < 1:
                    continue
                for anchor in range(episode_start, episode_end):
                    candidate_anchors.append((anchor, episode_end))

            if not candidate_anchors:
                continue

            choice_idx = int(torch.randint(len(candidate_anchors), (1,), device=self.device).item())
            anchor_step, episode_end = candidate_anchors[choice_idx]
            max_future = episode_end - anchor_step
            if max_future < 1:
                continue

            delta = int(geom_dist.sample().item())
            attempts = 0
            while delta > max_future and attempts < 10:
                delta = int(geom_dist.sample().item())
                attempts += 1
            if delta > max_future:
                delta = max_future

            positive_step = anchor_step + delta

            batch_indices.append(batch_idx)
            anchor_steps.append(anchor_step)
            positive_steps.append(positive_step)
            sampled_deltas.append(float(delta))

        num_pairs = len(batch_indices)
        if num_pairs < 2:
            return torch.tensor(0.0, device=self.device), {
                "positive_sim_mean": 0.0,
                "negative_sim_mean": 0.0,
                "positive_sim_std": 0.0,
                "negative_sim_std": 0.0,
                "num_pairs": num_pairs,
                "delta_mean": 0.0,
            }

        batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        anchor_idx_tensor = torch.tensor(anchor_steps, device=self.device, dtype=torch.long)
        positive_idx_tensor = torch.tensor(positive_steps, device=self.device, dtype=torch.long)

        anchor_embeddings = embeddings[batch_idx_tensor, anchor_idx_tensor]
        positive_embeddings = embeddings[batch_idx_tensor, positive_idx_tensor]

        similarities = anchor_embeddings @ positive_embeddings.T
        positive_logits = similarities.diagonal().unsqueeze(1)
        mask = torch.eye(num_pairs, device=self.device, dtype=torch.bool)
        negative_logits = similarities[~mask].view(num_pairs, num_pairs - 1)

        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.temperature
        labels = torch.zeros(num_pairs, dtype=torch.long, device=self.device)

        infonce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")

        positive_sim_mean = positive_logits.mean().item()
        negative_sim_mean = negative_logits.mean().item()
        positive_sim_std = positive_logits.std().item()
        negative_sim_std = negative_logits.std().item()
        delta_mean = float(sum(sampled_deltas) / len(sampled_deltas))

        metrics = {
            "positive_sim_mean": positive_sim_mean,
            "negative_sim_mean": negative_sim_mean,
            "positive_sim_std": positive_sim_std,
            "negative_sim_std": negative_sim_std,
            "num_pairs": num_pairs,
            "delta_mean": delta_mean,
        }

        return infonce_loss * self.contrastive_coef, metrics
