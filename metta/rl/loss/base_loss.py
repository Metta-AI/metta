from abc import ABC, abstractmethod
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.rl.loss.loss_tracker import LossTracker
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState


class BaseLoss(ABC):
    """
    The Loss class acts as a manager for different loss computations.

    It is initialized with the shared trainer state (policy, config, device, etc.)
    and dynamically instantiates the required loss components (e.g., PPO, Contrastive)
    based on the configuration. Each component holds a reference to this manager
    to access the shared state, favoring composition over inheritance.
    """

    __slots__ = (
        "policy",
        "policy_experience_spec",
        "trainer_cfg",
        "vec_env",
        "device",
        "loss_tracker",
        "policy_store",
        "policy_cfg",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
        policy_store: PolicyStore,
    ):
        self.policy = policy
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.trainer_cfg = trainer_cfg
        self.policy_cfg = self.policy.get_cfg()
        self.vec_env = vec_env
        self.device = device
        self.loss_tracker = loss_tracker
        self.policy_store = policy_store

        # populate loss tracker with the loss components
        # self.loss_tracker.add_loss_component(self)

    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience spec required by this loss.

        Defaults to an empty Composite, meaning the loss does not require
        additional fields beyond the policy's experience spec.
        """
        return Composite()

    # av fix
    # def roll_out(self) -> None:
    #     """Uses trainer to work with the env to generate experience."""
    #     while not self.policy.experience.full:
    #         # expecting trainer.roll_out() to get obs from env, run policy, get td from policy, populate with env attr
    #         # like rewards, dones, truncateds, etc., then for policy.experience to take what it wants from the td
    #         self.policy.trainer.roll_out()

    # av consider eliminating trainer_state
    @abstractmethod
    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Compute loss and write any shared minibatch data needed by other losses."""
        ...

    def on_mb_end(self):
        """For instance, allow losses with their own optimizers to run"""
        return

    def save_loss_states(self):
        """Save states to the policy."""
        return

    def losses_to_track(self) -> list[str]:
        """
        Declare metric keys this loss would like the trainer to track and report.

        Default returns an empty list for compatibility. Concrete losses can
        override to request specific keys (e.g., ["policy_loss", "value_loss"]).
        """
        return []

    # helper method for losses that wish to detach grads from tensors at various components in the policy
    # av delete this
    # def find_components_recursively(self, leaf: str, target: str) -> list[str]:
    #     """Recursively walk the MettaAgent and find the component names between a single leaf and a single target
    #     component. It includes the leaf but not the target in the list.
    #     Run this function for each leaf and target pair if necessary."""

    #     def _check_component_name(node: str, target: str, keys: list[str]) -> None:
    #         sources = getattr(self.policy.components[node], "_sources", None)
    #         if sources is None:
    #             return
    #         for source in sources:
    #             if source["name"] != target:
    #                 keys.append(source["name"])
    #                 _check_component_name(source["name"], target, keys)
    #             else:
    #                 keys.append(target)
    #                 return

    #     keys = []
    #     _check_component_name(leaf, target, keys)

    #     return keys
