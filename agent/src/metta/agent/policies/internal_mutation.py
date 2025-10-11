import logging
from typing import cast

import torch
from tensordict import TensorDict

from .fast import FastConfig, FastPolicy

logger = logging.getLogger(__name__)


class InternalMutationConfig(FastConfig):
    """Configuration identical to FastConfig, except we expand the action space
    to include 5 self-mutation actions (n1...n5)."""

    class_path: str = "metta.agent.policies.internal_mutation.InternalMutationPolicy"
    num_internal_actions: int = 5


class InternalMutationPolicy(FastPolicy):
    """Policy variant that adds self-modification actions (n1...n5).

    Internal actions:
        n1...n5 → Add ±(1-5)% random noise to all model parameters,
        then re-run forward() until a non-internal action is selected.
    """

    def __init__(self, env, config: InternalMutationConfig | None = None):
        super().__init__(env, config or InternalMutationConfig())

        self.config = cast(InternalMutationConfig, self.config)
        self.num_internal_actions = self.config.num_internal_actions
        self.env_action_dim = env.action_space.n
        self.total_action_dim = self.env_action_dim + self.num_internal_actions

        logger.info(
            f"Expanding action space: env={self.env_action_dim}, "
            f"internal={self.num_internal_actions}, total={self.total_action_dim}"
        )

    @torch._dynamo.disable
    def forward(
        self,
        td: TensorDict,
        state=None,
        action: torch.Tensor | None = None,
    ) -> TensorDict:
        """Forward pass with recursive self-mutation until an external action is chosen."""
        td = super().forward(td, state, action)

        # Determine selected action index (tensor → int)
        if "action" in td:
            selected_action = td["action"]
        elif action is not None:
            selected_action = action
        else:
            raise ValueError("No action found in TensorDict or provided explicitly.")

        if torch.is_tensor(selected_action):
            selected_action = selected_action.item()

        # While the selected action is internal, mutate + recompute
        while selected_action >= self.env_action_dim:
            internal_index = selected_action - self.env_action_dim
            mutation_strength = (internal_index + 1) * 0.01  # 1%-5%

            logger.info(f"[InternalMutation] Performing self-perturbation (±{mutation_strength * 100:.1f}%)")
            self._apply_mutation(mutation_strength)

            # Recalculate logits and resample action
            td = super().forward(td, state, None)
            if "action" in td:
                selected_action = td["action"]
                if torch.is_tensor(selected_action):
                    selected_action = selected_action.item()
            else:
                break  # Defensive fallback

        return td

    def _apply_mutation(self, magnitude: float):
        """Multiply weights by (1 + noise), noise ∈ [−magnitude, +magnitude]."""
        with torch.no_grad():
            for _name, param in self.named_parameters():
                if param.requires_grad:
                    noise = torch.empty_like(param).uniform_(-magnitude, magnitude)
                    param.mul_(1.0 + noise)
        logger.debug(f"Applied ±{magnitude * 100:.1f}% mutation to all parameters.")

    def initialize_to_environment(self, env, device):
        logs = super().initialize_to_environment(env, device)
        logs.append("InternalMutationPolicy initialized with internal mutation actions.")
        return logs
