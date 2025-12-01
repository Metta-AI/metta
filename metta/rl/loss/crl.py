# metta/rl/loss/crl.py
"""Contrastive Reinforcement Learning (CRL) Implementation.

Based on "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable
New Goal-Reaching Capabilities" (Wang et al., 2025).

CRL is a self-supervised goal-conditioned RL algorithm that uses contrastive learning
to train a temporal distance function (critic) and policy. Key features:

1. The critic learns to predict reachability via InfoNCE contrastive loss
2. The actor is trained to maximize the critic (reach commanded goals)
3. No explicit reward function - the critic output IS the reward signal
4. Supports very deep networks (up to 1024 layers) with residual connections

Usage:
    Enable CRL in your training config:
    ```yaml
    trainer:
      losses:
        crl:
          enabled: true
          depth: 64  # Network depth (must be multiple of 4)
          hidden_dim: 256
          embedding_dim: 64
    ```
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.loss.crl_critic import CRLCritic
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class CRLConfig(LossConfig):
    """Configuration for Contrastive RL.

    The algorithm trains a goal-conditioned policy using contrastive learning.
    The critic learns temporal distances via InfoNCE, and the actor maximizes these.
    """

    # Network architecture
    depth: int = Field(
        default=64,
        ge=4,
        description="Total depth (number of Dense layers) for critic encoders. Must be multiple of 4.",
    )
    hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dimension for residual blocks",
    )
    embedding_dim: int = Field(
        default=64,
        gt=0,
        description="Output embedding dimension for encoders",
    )
    action_embed_dim: int = Field(
        default=32,
        gt=0,
        description="Embedding dimension for discrete actions",
    )

    # Training hyperparameters
    temperature: float = Field(
        default=0.1,
        gt=0,
        description="Temperature for InfoNCE loss (τ in softmax)",
    )
    critic_lr: float = Field(
        default=3e-4,
        gt=0,
        description="Learning rate for critic networks",
    )
    critic_coef: float = Field(
        default=1.0,
        ge=0,
        description="Coefficient for critic loss",
    )

    # Goal sampling
    discount: float = Field(
        default=0.99,
        ge=0,
        lt=1,
        description="Discount factor γ for geometric positive sampling",
    )
    logsumexp_penalty: float = Field(
        default=0.1,
        ge=0,
        description="Coefficient for logsumexp regularization",
    )

    # Goal conditioning options
    goal_key: Optional[str] = Field(
        default=None,
        description="Key for goal observations. If None, uses state observations as goals.",
    )
    goal_dims: Optional[list[int]] = Field(
        default=None,
        description="Dimensions of state to use as goal (e.g., [0, 1] for xy position)",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "CRLLoss":
        return CRLLoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class CRLLoss(Loss):
    """Contrastive RL Loss for training the goal-conditioned critic.

    This loss trains the CRL critic networks using the InfoNCE objective:
    - Positive pairs: (s, a, g) where g is a future state from the same trajectory
    - Negative pairs: (s, a, g') where g' is from a different trajectory

    The critic learns a temporal distance function that can be used to derive rewards.
    """

    __slots__ = (
        "critic",
        "temperature",
        "critic_coef",
        "discount",
        "logsumexp_penalty",
        "goal_dims",
        "goal_key",
        "_state_dim",
        "_goal_dim",
        "_num_actions",
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

        self.temperature = self.cfg.temperature
        self.critic_coef = self.cfg.critic_coef
        self.discount = self.cfg.discount
        self.logsumexp_penalty = self.cfg.logsumexp_penalty
        self.goal_key = self.cfg.goal_key
        self.goal_dims = self.cfg.goal_dims

        # Infer dimensions from environment
        obs_space = env.single_observation_space
        action_space = env.single_action_space

        # State dimension from observation space
        if hasattr(obs_space, "shape"):
            # Flatten observation for state dimension
            self._state_dim = int(torch.tensor(obs_space.shape).prod().item())
        else:
            raise ValueError("Cannot infer state dimension from observation space")

        # Goal dimension - same as state or subset
        if self.goal_dims is not None:
            self._goal_dim = len(self.goal_dims)
        else:
            self._goal_dim = self._state_dim

        # Action space
        if hasattr(action_space, "n"):
            # Discrete action space
            self._num_actions = action_space.n
            discrete_actions = True
            action_dim = 1
        else:
            # Continuous action space
            self._num_actions = None
            discrete_actions = False
            action_dim = int(torch.tensor(action_space.shape).prod().item())

        # Create critic network
        self.critic = CRLCritic(
            state_dim=self._state_dim,
            action_dim=action_dim,
            goal_dim=self._goal_dim,
            hidden_dim=self.cfg.hidden_dim,
            embedding_dim=self.cfg.embedding_dim,
            depth=self.cfg.depth,
            discrete_actions=discrete_actions,
            num_actions=self._num_actions,
            action_embed_dim=self.cfg.action_embed_dim,
        ).to(device)

        # Register critic parameters with the loss for state dict
        self.register_state_attr("critic")

    def get_experience_spec(self) -> Composite:
        """Define additional data needed for CRL."""
        return Composite(
            # Store flat observations for goal sampling
            crl_obs=UnboundedContinuous(shape=torch.Size([self._state_dim]), dtype=torch.float32),
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Store flattened observations for goal sampling during training."""
        # Extract and flatten observations for goal sampling
        if "env_obs" in td.keys():
            obs = td["env_obs"]
            # Flatten observation to vector
            flat_obs = obs.reshape(obs.shape[0], -1).float()
            td["crl_obs"] = flat_obs

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute CRL InfoNCE loss for critic training."""
        minibatch = shared_loss_data["sampled_mb"]
        batch_shape = minibatch.batch_size

        if len(batch_shape) != 2:
            raise ValueError("CRL loss expects minibatch with 2D batch size (segments, horizon)")

        segments, horizon = batch_shape

        # Get observations and actions
        if "crl_obs" in minibatch.keys():
            obs = minibatch["crl_obs"]  # (segments, horizon, state_dim)
        else:
            # Fallback to env_obs
            obs = minibatch["env_obs"].reshape(segments, horizon, -1).float()

        actions = minibatch["actions"]  # (segments, horizon) or (segments, horizon, action_dim)

        # Get done masks for episode boundaries
        dones = minibatch.get("dones", None)
        if dones is not None:
            dones = dones.squeeze(-1) if dones.dim() == 3 else dones
            done_mask = dones.to(dtype=torch.bool)
        else:
            done_mask = torch.zeros(segments, horizon, dtype=torch.bool, device=self.device)

        truncateds = minibatch.get("truncateds", None)
        if truncateds is not None:
            truncateds = truncateds.squeeze(-1) if truncateds.dim() == 3 else truncateds
            done_mask = torch.logical_or(done_mask, truncateds.to(dtype=torch.bool))

        # Sample positive and negative pairs for InfoNCE
        infonce_loss, metrics = self._compute_infonce_loss(
            obs, actions, done_mask, segments, horizon
        )

        # Track metrics
        self.loss_tracker["crl_critic_loss"].append(float(infonce_loss.item()))
        for key, value in metrics.items():
            if key not in self.loss_tracker:
                self.loss_tracker[key] = []
            self.loss_tracker[key].append(float(value))

        # Store critic for actor loss
        shared_loss_data["crl_critic"] = self.critic

        return infonce_loss * self.critic_coef, shared_loss_data, False

    def _compute_infonce_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        done_mask: Tensor,
        segments: int,
        horizon: int,
    ) -> tuple[Tensor, dict]:
        """Compute InfoNCE contrastive loss.

        For each (s, a) pair, we sample a future state g from the same trajectory
        as the positive example, and use states from other trajectories as negatives.

        Args:
            obs: Observations of shape (segments, horizon, state_dim)
            actions: Actions of shape (segments, horizon) or (segments, horizon, action_dim)
            done_mask: Episode boundaries of shape (segments, horizon)
            segments: Number of trajectory segments
            horizon: Length of each segment

        Returns:
            loss: Scalar InfoNCE loss
            metrics: Dictionary of tracking metrics
        """
        # Sample positive pairs using geometric distribution
        prob = max(1.0 - float(self.discount), 1e-8)
        geom_dist = torch.distributions.Geometric(
            probs=torch.tensor(prob, device=self.device, dtype=obs.dtype)
        )

        done_mask_cpu = done_mask.detach().cpu()

        batch_indices: list[int] = []
        anchor_steps: list[int] = []
        positive_steps: list[int] = []
        sampled_deltas: list[float] = []

        # Sample one positive pair per segment
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

            # Find valid anchor-positive pairs within episodes
            candidate_anchors: list[tuple[int, int]] = []
            for episode_start, episode_end in episode_bounds:
                if episode_end - episode_start < 1:
                    continue
                for anchor in range(episode_start, episode_end):
                    candidate_anchors.append((anchor, episode_end))

            if not candidate_anchors:
                continue

            # Randomly select an anchor
            choice_idx = int(torch.randint(len(candidate_anchors), (1,), device=self.device).item())
            anchor_step, episode_end = candidate_anchors[choice_idx]
            max_future = episode_end - anchor_step

            if max_future < 1:
                continue

            # Sample delta from geometric distribution
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
                "crl_positive_sim": 0.0,
                "crl_negative_sim": 0.0,
                "crl_num_pairs": num_pairs,
                "crl_delta_mean": 0.0,
            }

        # Convert to tensors
        batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        anchor_idx_tensor = torch.tensor(anchor_steps, device=self.device, dtype=torch.long)
        positive_idx_tensor = torch.tensor(positive_steps, device=self.device, dtype=torch.long)

        # Get anchor states and actions
        anchor_states = obs[batch_idx_tensor, anchor_idx_tensor]  # (N, state_dim)
        anchor_actions = actions[batch_idx_tensor, anchor_idx_tensor]  # (N,) or (N, action_dim)

        # Get positive goal states
        positive_goals = obs[batch_idx_tensor, positive_idx_tensor]  # (N, state_dim)

        # Apply goal_dims if specified
        if self.goal_dims is not None:
            positive_goals = positive_goals[..., self.goal_dims]

        # Encode state-action pairs
        phi = self.critic.encode_state_action(anchor_states, anchor_actions)  # (N, embedding_dim)

        # Encode all positive goals (they become negatives for other samples)
        psi = self.critic.encode_goal(positive_goals)  # (N, embedding_dim)

        # Compute pairwise distances
        # For InfoNCE, we use all goals as candidates for each state-action pair
        # Positive is on the diagonal
        distances = torch.cdist(phi, psi, p=2)  # (N, N)

        # Convert to logits (negative distance / temperature)
        logits = -distances / self.temperature

        # Labels: each sample's positive is on the diagonal
        labels = torch.arange(num_pairs, device=self.device, dtype=torch.long)

        # InfoNCE loss (cross-entropy with positive on diagonal)
        infonce_loss = F.cross_entropy(logits, labels, reduction="mean")

        # Add logsumexp regularization to prevent collapse
        if self.logsumexp_penalty > 0:
            logsumexp_reg = torch.logsumexp(logits, dim=1).mean()
            infonce_loss = infonce_loss + self.logsumexp_penalty * logsumexp_reg

        # Compute metrics
        with torch.no_grad():
            positive_logits = logits.diagonal()
            mask = ~torch.eye(num_pairs, dtype=torch.bool, device=self.device)
            negative_logits = logits[mask].view(num_pairs, num_pairs - 1)

            positive_sim = positive_logits.mean().item()
            negative_sim = negative_logits.mean().item()
            delta_mean = sum(sampled_deltas) / len(sampled_deltas)

        metrics = {
            "crl_positive_sim": positive_sim,
            "crl_negative_sim": negative_sim,
            "crl_num_pairs": num_pairs,
            "crl_delta_mean": delta_mean,
        }

        return infonce_loss, metrics

    def state_dict(self):
        """Return state dict including critic parameters."""
        state = super().state_dict()
        state["critic_state_dict"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state_dict, *, strict: bool = True):
        """Load state dict including critic parameters."""
        if "critic_state_dict" in state_dict:
            self.critic.load_state_dict(state_dict["critic_state_dict"])
            # Remove from dict before parent handles it
            state_dict = {k: v for k, v in state_dict.items() if k != "critic_state_dict"}
        return super().load_state_dict(state_dict, strict=strict)
