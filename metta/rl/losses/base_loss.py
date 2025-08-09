from abc import ABC, abstractmethod
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from metta.agent.metta_agent import PolicyAgent
from metta.rl.trainer_state import TrainerState


class LossTracker:
    def __init__(self):
        self.zero()

    def zero(self):
        """Reset all loss values to 0.0."""
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.approx_kl_sum = 0.0
        self.clipfrac_sum = 0.0
        self.l2_reg_loss_sum = 0.0
        self.l2_init_loss_sum = 0.0
        self.ks_action_loss_sum = 0.0
        self.ks_value_loss_sum = 0.0
        self.importance_sum = 0.0
        self.current_logprobs_sum = 0.0
        self.explained_variance = 0.0
        self.minibatches_processed = 0

    def stats(self) -> dict[str, float]:
        """Convert losses to dictionary with proper averages."""
        n = max(1, self.minibatches_processed)

        return {
            "policy_loss": self.policy_loss_sum / n,
            "value_loss": self.value_loss_sum / n,
            "entropy": self.entropy_sum / n,
            "approx_kl": self.approx_kl_sum / n,
            "clipfrac": self.clipfrac_sum / n,
            "l2_reg_loss": self.l2_reg_loss_sum / n,
            "l2_init_loss": self.l2_init_loss_sum / n,
            "ks_action_loss": self.ks_action_loss_sum / n,
            "ks_value_loss": self.ks_value_loss_sum / n,
            "importance": self.importance_sum / n,
            "explained_variance": self.explained_variance,
            "current_logprobs": self.current_logprobs_sum / n,
        }


class BaseLoss(ABC):
    """
    The Loss class acts as a manager for different loss computations.

    It is initialized with the shared trainer state (policy, config, device, etc.)
    and dynamically instantiates the required loss components (e.g., PPO, Contrastive)
    based on the configuration. Each component holds a reference to this manager
    to access the shared state, favoring composition over inheritance.
    """

    def __init__(
        self,
        policy: PolicyAgent,
        cfg: Any,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
    ):
        self.policy = policy
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.cfg = cfg
        self.vec_env = vec_env
        self.device = device
        self.loss_tracker = loss_tracker

        # populate loss tracker with the loss components
        # self.loss_tracker.add_loss_component(self)

    @abstractmethod
    def get_experience_spec(self) -> TensorDict:
        pass

    # av fix
    # def roll_out(self) -> None:
    #     """Uses trainer to work with the env to generate experience."""
    #     while not self.policy.experience.full:
    #         # expecting trainer.roll_out() to get obs from env, run policy, get td from policy, populate with env attr
    #         # like rewards, dones, truncateds, etc., then for policy.experience to take what it wants from the td
    #         self.policy.trainer.roll_out()

    @abstractmethod
    # av consider eliminating trainer_state
    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """This is primarily computing loss and feeding to the optimizer."""
        pass

    # helper method for losses that wish to detach grads from tensors at various components in the policy
    def find_components_recursively(self, leaf: str, target: str) -> list[str]:
        """Recursively walk the MettaAgent and find the component names between a single leaf and a single target
        component. It includes the leaf but not the target in the list.
        Run this function for each leaf and target pair if necessary."""

        def _check_component_name(node: str, target: str, keys: list[str]) -> None:
            sources = getattr(self.policy.components[node], "_sources", None)
            if sources is None:
                return
            for source in sources:
                if source["name"] != target:
                    keys.append(source["name"])
                    _check_component_name(source["name"], target, keys)
                else:
                    keys.append(target)
                    return

        keys = []
        _check_component_name(leaf, target, keys)

        return keys
