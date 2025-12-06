"""Inverse Dynamics Model for action-relevant representation learning.

Predicts the action taken between two consecutive states. This forces
the encoder to capture features that are relevant to agent actions,
filtering out task-irrelevant distractors.

Reference: Pathak et al., 2017 "Curiosity-driven Exploration by Self-Supervised Prediction"
https://arxiv.org/abs/1705.05363
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class InverseDynamicsConfig(LossConfig):
    """Configuration for Inverse Dynamics loss."""

    # Coefficient for inverse dynamics loss
    inv_dyn_coef: float = Field(default=0.1, ge=0.0)

    # Hidden dimension for inverse dynamics predictor
    hidden_dim: int = Field(default=256, gt=0)

    # Whether to use the encoder output or raw observations
    use_encoder_features: bool = True

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "InverseDynamics":
        return InverseDynamics(policy, trainer_cfg, env, device, instance_name, self)


class InverseDynamicsPredictor(nn.Module):
    """Predicts action from (s_t, s_{t+1}) pair."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        # Input is concatenation of two state embeddings
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, s_t: Tensor, s_t1: Tensor) -> Tensor:
        """Predict action logits from state pair."""
        combined = torch.cat([s_t, s_t1], dim=-1)
        return self.net(combined)


class InverseDynamics(Loss):
    """Inverse Dynamics loss for learning action-relevant features.

    Given encoder features for s_t and s_{t+1}, predicts the action a_t
    that was taken. This encourages the encoder to capture features
    that are predictive of actions, filtering out distractors.
    """

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: InverseDynamicsConfig,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)

        # Get number of actions
        self._num_actions = env.single_action_space.n

        # Lazy init predictor
        self._input_dim: int | None = None
        self._predictor: InverseDynamicsPredictor | None = None

    def _initialize_predictor(self, input_dim: int) -> None:
        """Initialize predictor network."""
        if self._input_dim == input_dim:
            return

        self._input_dim = input_dim
        self._predictor = InverseDynamicsPredictor(input_dim, self.cfg.hidden_dim, self._num_actions).to(self.device)

    def get_experience_spec(self) -> Composite:
        """No additional experience fields needed."""
        return Composite()

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """No-op during rollout."""
        pass

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Train inverse dynamics predictor."""
        cfg = self.cfg

        policy_td = shared_loss_data.get("policy_td")
        sampled_mb = shared_loss_data.get("sampled_mb")

        if policy_td is None or sampled_mb is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), shared_loss_data, False

        if "core" not in policy_td.keys():
            return torch.tensor(0.0, device=self.device, requires_grad=True), shared_loss_data, False

        # Get encoder features - shape (B, T, D)
        core = policy_td["core"]
        if core.dim() == 2:
            # No temporal dimension, can't do inverse dynamics
            return torch.tensor(0.0, device=self.device, requires_grad=True), shared_loss_data, False

        B, T, D = core.shape
        if T < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True), shared_loss_data, False

        # Initialize predictor
        self._initialize_predictor(D)

        # Get consecutive state pairs
        s_t = core[:, :-1, :].reshape(-1, D)  # (B*(T-1), D)
        s_t1 = core[:, 1:, :].reshape(-1, D)  # (B*(T-1), D)

        # Get actions taken between s_t and s_t1
        actions = sampled_mb["actions"]
        if actions.dim() == 2:
            actions = actions[:, :-1].reshape(-1)  # (B*(T-1),)
        else:
            actions = actions[:-1]

        # Ensure shapes match
        min_len = min(s_t.shape[0], actions.shape[0])
        s_t = s_t[:min_len]
        s_t1 = s_t1[:min_len]
        actions = actions[:min_len].long()

        # Predict actions
        action_logits = self._predictor(s_t, s_t1)

        # Cross-entropy loss
        inv_dyn_loss = F.cross_entropy(action_logits, actions)
        weighted_loss = cfg.inv_dyn_coef * inv_dyn_loss

        # Compute accuracy for logging
        with torch.no_grad():
            predicted_actions = action_logits.argmax(dim=-1)
            accuracy = (predicted_actions == actions).float().mean()

        # Track metrics
        self._track("inv_dyn_loss", inv_dyn_loss)
        self._track("inv_dyn_accuracy", accuracy)

        return weighted_loss, shared_loss_data, False

    def _track(self, key: str, value: Tensor) -> None:
        if value.numel() == 1:
            self.loss_tracker[key].append(float(value.item()))
        else:
            self.loss_tracker[key].append(float(value.mean().item()))
