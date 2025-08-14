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


class BaseLoss:
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
        "loss_cfg",
        "rollout_start_epoch",
        "rollout_end_epoch",
        "train_start_epoch",
        "train_end_epoch",
        "instance_name",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: TrainerConfig,
        vec_env: Any,
        device: torch.device,
        loss_tracker: LossTracker,
        policy_store: PolicyStore,
        instance_name: str,
    ):
        self.policy = policy
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.trainer_cfg = trainer_cfg
        self.policy_cfg = self.policy.get_cfg()
        self.vec_env = vec_env
        self.device = device
        self.loss_tracker = loss_tracker
        self.policy_store = policy_store
        self.instance_name = instance_name

        self.loss_cfg = self.policy_cfg.losses.get(self.instance_name, {})

        # Get schedule for rollout and train
        schedule_cfg = self.loss_cfg.get("schedule") or {}

        rollout_cfg = schedule_cfg.get("rollout") or {}
        self.rollout_start_epoch = rollout_cfg.get("begin_at_epoch", 0)
        self.rollout_end_epoch = rollout_cfg.get("end_at_epoch", float("inf"))

        train_cfg = schedule_cfg.get("train") or {}
        self.train_start_epoch = train_cfg.get("begin_at_epoch", 0)
        self.train_end_epoch = train_cfg.get("end_at_epoch", float("inf"))

    # --- Control flow ---
    def on_new_training_run(self) -> None:
        """We're at the very beginning of the training loop."""
        self.policy.on_new_training_run()
        return

    def on_rollout_start(self) -> None:
        self.policy.on_rollout_start()
        return

    def on_training_phase_start(self) -> None:
        """We've completed the rollout phase and are starting the train phase."""
        return

    def should_run_rollout(self, epoch: int) -> bool:
        """Whether this loss should run its rollout phase, based on the current agent step."""
        return self.rollout_start_epoch <= epoch < self.rollout_end_epoch

    def should_run_train(self, epoch: int) -> bool:
        """Whether this loss should run its train phase, based on the current agent step."""
        return self.train_start_epoch <= epoch < self.train_end_epoch

    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience spec required by this loss."""
        return Composite()

    # BaseLoss handles the logic for running rollout and train phases, keeping super loss class simple
    # but this might be more confusing for new researchers and takes control away from the super loss
    def rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        if not self.should_run_rollout(trainer_state.epoch):
            return
        self.run_rollout(td, trainer_state)

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        """Override this method in subclasses to implement rollout logic."""
        return

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Compute loss and write any shared minibatch data needed by other losses."""
        if not self.should_run_train(trainer_state.epoch):
            return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data
        return self.run_train(shared_loss_data, trainer_state)

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Override this method in subclasses to implement train logic."""
        return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data

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
