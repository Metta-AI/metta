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
        "rollout_cycle_length",
        "rollout_active_in_cycle",
        "train_cycle_length",
        "train_active_in_cycle",
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

        self._get_schedule()

    # ----------- INITIALIZE REPLAY BUFFER AND LOSS TRACKER -----------
    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience spec required by this loss."""
        return Composite()

    def losses_to_track(self) -> list[str]:
        """
        Declare metric keys this loss would like the trainer to track and report.

        Default returns an empty list for compatibility. Concrete losses can
        override to request specific keys (e.g., ["policy_loss", "value_loss"]).
        """
        return []

    # ----------- END INITIALIZE REPLAY BUFFER AND LOSS TRACKER -----------

    # ======================================================================
    # ============================ CONTROL FLOW ============================
    # BaseLoss provides defaults for every control flow method and even handles the scheduling logic. Simply override
    # any of these methods in your Loss class to implement your own logic when needed.

    def on_new_training_run(self) -> None:
        """We're at the very beginning of the training loop."""
        self.policy.on_new_training_run()
        return

    def on_rollout_start(self) -> None:
        """We're about to start a new rollout phase."""
        self.policy.on_rollout_start()
        return

    def rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        """Repeatedly called rollout steps until you set completion.
        Each step gets obs and returns actions to the env."""
        if not self._should_run_rollout(trainer_state.epoch):
            return
        self.run_rollout(td, trainer_state)

    def run_rollout(self, td: TensorDict, trainer_state: TrainerState) -> None:
        """Override this method in subclasses to implement rollout logic. Or override rollout() if you need to override
        the scheduling logic."""
        return

    def on_training_phase_start(self) -> None:
        """We've completed the rollout phase and are starting the train phase."""
        return

    def train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Repeatedly called training steps until the total number of minibatches (set in cfg) is reached.
        Compute loss and write any shared minibatch data needed by other losses."""
        if not self._should_run_train(trainer_state.epoch):
            return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data
        return self.run_train(shared_loss_data, trainer_state)

    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        """Override this method in subclasses to implement train logic. Or override train() if you need to override
        the scheduling logic."""
        return torch.tensor(0.0, device=self.device, dtype=torch.float32), shared_loss_data

    def on_mb_end(self):
        """For instance, allow losses with their own optimizers to run"""
        return

    def on_train_phase_end(self) -> None:
        """We've completed the train phase and will be transitioning to the next rollout phase."""
        return

    def save_loss_states(self):
        # TODO: Implement this
        """Save loss states at the end of the training run in case you need to resume training later. This is currentnly
        not implemented."""
        return

    # ---------------- Internal Scheduling Logic for Rollout and Train ----------------
    def _should_run_rollout(self, epoch: int) -> bool:
        """Whether this loss should run its rollout phase, based on the current agent step."""
        in_range = self.rollout_start_epoch <= epoch < self.rollout_end_epoch
        if not in_range:
            return False

        if self.rollout_cycle_length is not None:
            if not self.rollout_active_in_cycle:
                return False

            # Assuming epoch is 0-indexed. User config is 1-indexed.
            epoch_in_cycle = (epoch % self.rollout_cycle_length) + 1
            return epoch_in_cycle in self.rollout_active_in_cycle

        return True

    def _should_run_train(self, epoch: int) -> bool:
        """Whether this loss should run its train phase, based on the current agent step."""
        in_range = self.train_start_epoch <= epoch < self.train_end_epoch
        if not in_range:
            return False

        if self.train_cycle_length is not None:
            if not self.train_active_in_cycle:
                return False

            epoch_in_cycle = (epoch % self.train_cycle_length) + 1
            return epoch_in_cycle in self.train_active_in_cycle

        return True

    # ---------------- END Internal Scheduling Logic for Rollout and Train ----------------

    # ============================ END CONTROL FLOW ============================
    # ==========================================================================

    def _get_schedule(self):
        """Helper for initializing variables used in scheduling logic."""
        schedule_cfg = self.loss_cfg.get("schedule") or {}

        rollout_cfg = schedule_cfg.get("rollout") or {}
        self.rollout_start_epoch = rollout_cfg.get("begin_at_epoch", 0)
        self.rollout_end_epoch = rollout_cfg.get("end_at_epoch", float("inf"))
        self.rollout_cycle_length = rollout_cfg.get("cycle_length")
        self.rollout_active_in_cycle = rollout_cfg.get("active_in_cycle")

        train_cfg = schedule_cfg.get("train") or {}
        self.train_start_epoch = train_cfg.get("begin_at_epoch", 0)
        self.train_end_epoch = train_cfg.get("end_at_epoch", float("inf"))
        self.train_cycle_length = train_cfg.get("cycle_length")
        self.train_active_in_cycle = train_cfg.get("active_in_cycle")
