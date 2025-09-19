from collections import defaultdict
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training.context import TrainerContext
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment


class Loss:
    """Base class coordinating rollout and training behaviour for concrete losses."""

    __slots__ = (
        "policy",
        "replay",
        "policy_experience_spec",
        "trainer_cfg",
        "env",
        "device",
        "loss_tracker",
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
        "_context",
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
        self.policy = policy
        self.trainer_cfg = trainer_cfg
        self.env = env
        self.device = device
        self.instance_name = instance_name
        self.loss_cfg = loss_config
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.loss_tracker = defaultdict(list)
        self._context: TrainerContext | None = None

        self._get_schedule()

    def attach_context(self, context: TrainerContext) -> None:
        """Register the shared trainer context for this loss instance."""
        self._context = context

    def _require_context(self, context: TrainerContext | None = None) -> TrainerContext:
        if context is not None:
            self._context = context
            return context
        if self._context is None:
            raise RuntimeError("Loss has not been attached to a TrainerContext")
        return self._context

    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience replay buffer spec required by this loss."""
        return Composite()

    # --------- Control flow hooks; override in subclasses when custom behaviour is needed ---------

    def on_new_training_run(self, context: TrainerContext | None = None) -> None:
        """Called at the very beginning of a training epoch."""
        self._require_context(context)

    def on_rollout_start(self, context: TrainerContext | None = None) -> None:
        """Called before starting a rollout phase."""
        self._require_context(context)
        self.policy.reset_memory()

    def rollout(self, td: TensorDict, context: TrainerContext | None = None) -> None:
        """Rollout step executed while experience buffer requests more data."""
        ctx = self._require_context(context)
        if not self._should_run_rollout(ctx.epoch):
            return
        if ctx.training_env_id is None:
            raise RuntimeError("TrainerContext.training_env_id must be set before calling Loss.rollout")
        self.run_rollout(td, ctx)

    def run_rollout(self, td: TensorDict, context: TrainerContext) -> None:
        """Override in subclasses to implement rollout logic."""
        return

    def train(
        self,
        shared_loss_data: TensorDict,
        context: TrainerContext | None,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Training step executed while scheduler allows it."""
        ctx = self._require_context(context)
        if not self._should_run_train(ctx.epoch):
            zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            return zero, shared_loss_data, False
        return self.run_train(shared_loss_data, ctx, mb_idx)

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: TrainerContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Override in subclasses to implement training logic."""
        zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return zero, shared_loss_data, False

    def on_mb_end(self, context: TrainerContext | None, mb_idx: int) -> None:
        """Hook executed at the end of each minibatch."""
        self._require_context(context)

    def on_train_phase_end(self, context: TrainerContext | None = None) -> None:
        """Hook executed after the training phase completes."""
        self._require_context(context)

    def save_loss_states(self, context: TrainerContext | None = None) -> None:
        """Save loss states at the end of training (optional)."""
        self._require_context(context)

    # Scheduling helpers
    def _should_run_rollout(self, epoch: int) -> bool:
        """Whether this loss should run its rollout phase, based on the current epoch."""
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
        """Whether this loss should run its train phase, based on the current epoch."""
        in_range = self.train_start_epoch <= epoch < self.train_end_epoch
        if not in_range:
            return False

        if self.train_cycle_length is not None:
            if not self.train_active_in_cycle:
                return False

            epoch_in_cycle = (epoch % self.train_cycle_length) + 1
            return epoch_in_cycle in self.train_active_in_cycle

        return True

    # End scheduling helpers

    def _get_schedule(self) -> None:
        """Helper for initializing variables used in scheduling logic."""
        schedule_cfg = {}  # TODO: support self.loss_cfg.schedule when available

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

    # Utility helpers

    def stats(self) -> dict[str, float]:
        """Aggregate tracked statistics into mean values."""
        return {k: sum(v) / len(v) if v else 0.0 for k, v in self.loss_tracker.items()}

    def zero_loss_tracker(self) -> None:
        """Zero all values in the loss tracker."""
        for key in self.loss_tracker.keys():
            self.loss_tracker[key].clear()

    def attach_replay_buffer(self, experience: Experience) -> None:
        """Attach the replay buffer to the loss."""
        self.replay = experience

    # End utility helpers
