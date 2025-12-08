"""Random Network Distillation (RND) for curiosity-driven exploration.

RND provides intrinsic rewards for novel states by measuring prediction error
between a fixed random target network and a learned predictor network.

High prediction error = novel state = intrinsic reward bonus

Reference: Burda et al., 2018 "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894
"""

from typing import Any

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class RNDConfig(LossConfig):
    """Configuration for Random Network Distillation loss."""

    # Coefficient for RND prediction loss (trains predictor network)
    rnd_coef: float = Field(default=0.1, ge=0.0)

    # Coefficient for intrinsic reward added to extrinsic reward
    intrinsic_reward_coef: float = Field(default=0.01, ge=0.0)

    # Hidden dimension for target/predictor networks
    hidden_dim: int = Field(default=256, gt=0)

    # Output dimension for target/predictor networks
    output_dim: int = Field(default=64, gt=0)

    # Whether to normalize intrinsic rewards
    normalize_intrinsic: bool = True

    # Running mean/std update rate for normalization
    intrinsic_gamma: float = Field(default=0.99, ge=0, le=1)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "RND":
        return RND(policy, trainer_cfg, env, device, instance_name, self)


class RNDTargetNetwork(nn.Module):
    """Fixed random network - never trained."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Learned predictor network - trained to match target."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RND(Loss):
    """Random Network Distillation loss for exploration.

    Computes intrinsic reward as prediction error between fixed target
    network and learned predictor network. Novel states have high
    prediction error, encouraging exploration.
    """

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: RNDConfig,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)

        # Get input dimension from policy's core output
        # Lazy init - will be set on first forward
        self._input_dim: int | None = None
        self._target_net: RNDTargetNetwork | None = None
        self._predictor_net: RNDPredictorNetwork | None = None

        # Running statistics for intrinsic reward normalization
        self._intrinsic_mean = torch.tensor(0.0, device=device)
        self._intrinsic_var = torch.tensor(1.0, device=device)
        self._intrinsic_count = torch.tensor(1e-4, device=device)

    def _initialize_networks(self, input_dim: int) -> None:
        """Initialize target and predictor networks."""
        if self._input_dim == input_dim:
            return

        self._input_dim = input_dim
        cfg = self.cfg

        self._target_net = RNDTargetNetwork(input_dim, cfg.hidden_dim, cfg.output_dim).to(self.device)
        self._predictor_net = RNDPredictorNetwork(input_dim, cfg.hidden_dim, cfg.output_dim).to(self.device)

        # Register predictor params with optimizer (target stays frozen)
        # The predictor will be optimized via the loss

    def get_experience_spec(self) -> Composite:
        """RND doesn't require additional experience fields - computed during training."""
        return Composite()

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """No-op during rollout - RND loss computed during training."""
        pass

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Train predictor network to match target network."""
        cfg = self.cfg

        policy_td = shared_loss_data.get("policy_td")
        if policy_td is None or "core" not in policy_td.keys():
            return torch.tensor(0.0, device=self.device, requires_grad=True), shared_loss_data, False

        core = policy_td["core"]
        if core.dim() > 2:
            core = core.reshape(-1, core.shape[-1])

        # Initialize networks if needed
        self._initialize_networks(core.shape[-1])

        # Compute prediction loss
        with torch.no_grad():
            target_features = self._target_net(core)

        predicted_features = self._predictor_net(core)
        prediction_loss = ((predicted_features - target_features) ** 2).mean()

        weighted_loss = cfg.rnd_coef * prediction_loss

        # Track metrics
        self._track("rnd_loss", prediction_loss)

        # Also compute and track current intrinsic reward stats
        with torch.no_grad():
            intrinsic = self._compute_intrinsic_reward(core)
            self._track("rnd_intrinsic_mean", intrinsic.mean())
            self._track("rnd_intrinsic_std", intrinsic.std())

        return weighted_loss, shared_loss_data, False

    def _compute_intrinsic_reward(self, core: Tensor) -> Tensor:
        """Compute intrinsic reward as prediction error."""
        target_features = self._target_net(core)
        predicted_features = self._predictor_net(core)

        # Per-sample MSE as intrinsic reward
        intrinsic = ((predicted_features - target_features) ** 2).mean(dim=-1)

        # Normalize intrinsic rewards
        if self.cfg.normalize_intrinsic:
            intrinsic = self._normalize_intrinsic(intrinsic)

        return intrinsic * self.cfg.intrinsic_reward_coef

    def _normalize_intrinsic(self, intrinsic: Tensor) -> Tensor:
        """Normalize intrinsic rewards using running statistics."""
        # Update running stats
        batch_mean = intrinsic.mean()
        batch_var = intrinsic.var()
        batch_count = intrinsic.numel()

        gamma = self.cfg.intrinsic_gamma
        self._intrinsic_mean = gamma * self._intrinsic_mean + (1 - gamma) * batch_mean
        self._intrinsic_var = gamma * self._intrinsic_var + (1 - gamma) * batch_var
        self._intrinsic_count = gamma * self._intrinsic_count + (1 - gamma) * batch_count

        # Normalize
        return (intrinsic - self._intrinsic_mean) / (self._intrinsic_var.sqrt() + 1e-8)

    def _track(self, key: str, value: Tensor) -> None:
        if value.numel() == 1:
            self.loss_tracker[key].append(float(value.item()))
        else:
            self.loss_tracker[key].append(float(value.mean().item()))
