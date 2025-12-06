"""Diversity loss for encouraging representational exploration.

This loss works in conjunction with DiversityInjection component to automatically
increase exploration when policy gradients vanish. The key insight is that when
PPO loss → 0 (stuck in local minima), the diversity loss term dominates, pushing
α higher and increasing representational spread across agents.
"""

from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class DiversityLossConfig(LossConfig):
    """Configuration for diversity loss."""

    # Coefficient for diversity loss term (-log_alpha)
    # Start small (~0.01) and tune as needed
    diversity_coef: float = Field(default=0.01, ge=0)

    # Name of the DiversityInjection component in the policy
    # Used to find the log_alpha parameter
    diversity_component_name: str = "diversity_injection"

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "DiversityLoss":
        return DiversityLoss(policy, trainer_cfg, env, device, instance_name, self)


class DiversityLoss(Loss):
    """Diversity loss that encourages exploration when policy gradients vanish.

    Loss = -diversity_coef * log_alpha

    When α is small (low diversity), log_alpha is negative, so -log_alpha is positive
    and this loss encourages α to grow. When PPO loss is meaningful, its gradients
    dominate and α stays controlled. When stuck (PPO loss ≈ 0), diversity loss
    dominates and α grows, increasing representational spread.
    """

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: DiversityLossConfig,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self._diversity_component = None
        self._find_diversity_component()

    def _find_diversity_component(self) -> None:
        """Find the DiversityInjection component in the policy."""
        if hasattr(self.policy, "components"):
            component_name = self.cfg.diversity_component_name
            if component_name in self.policy.components:
                self._diversity_component = self.policy.components[component_name]
            else:
                # Try to find any DiversityInjection component
                from metta.agent.components.diversity_injection import DiversityInjection

                for _, component in self.policy.components.items():
                    if isinstance(component, DiversityInjection):
                        self._diversity_component = component
                        break

    def get_experience_spec(self) -> Composite:
        """Diversity loss doesn't require additional experience fields."""
        return Composite()

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """No-op during rollout - diversity loss only affects training."""
        pass

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute diversity loss from the DiversityInjection component."""
        if self._diversity_component is None:
            # No diversity component found, return zero loss
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero_loss, shared_loss_data, False

        # Get diversity loss from component
        diversity_loss = self._diversity_component.get_diversity_loss()
        weighted_loss = self.cfg.diversity_coef * diversity_loss

        # Track metrics
        alpha = self._diversity_component.alpha
        self._track("diversity_loss", weighted_loss)
        self._track("diversity_alpha", alpha)
        self._track("diversity_log_alpha", self._diversity_component.log_alpha)

        return weighted_loss, shared_loss_data, False

    def _track(self, key: str, value: Tensor) -> None:
        """Track a metric value."""
        if value.numel() == 1:
            self.loss_tracker[key].append(float(value.item()))
        else:
            self.loss_tracker[key].append(float(value.mean().item()))
