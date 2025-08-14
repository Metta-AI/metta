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
        "rollout_start_step",
        "rollout_end_step",
        "train_start_step",
        "train_end_step",
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

        loss_name = self.__class__.__name__
        loss_cfg = self.policy_cfg.losses.get(loss_name, {})

        # Get schedule for rollout
        rollout_schedule = loss_cfg.get("rollout", {})
        self.rollout_start_step = rollout_schedule.get("start_step", 0)
        self.rollout_end_step = rollout_schedule.get("end_step", float("inf"))

        # Get schedule for train
        train_schedule = loss_cfg.get("train", {})
        self.train_start_step = train_schedule.get("start_step", 0)
        self.train_end_step = train_schedule.get("end_step", float("inf"))

    def should_run_rollout(self, agent_step: int) -> bool:
        """Whether this loss should run its rollout phase, based on the current agent step."""
        return self.rollout_start_step <= agent_step < self.rollout_end_step

    def should_run_train(self, agent_step: int) -> bool:
        """Whether this loss should run its train phase, based on the current agent step."""
        return self.train_start_step <= agent_step < self.train_end_step

    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience spec required by this loss."""
        return Composite()

    def rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        if not self.should_run_rollout(trainer_state.agent_step):
            return
        self.run_rollout(td, trainer_state)

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        """Override this method in subclasses to implement rollout logic."""
        return

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
