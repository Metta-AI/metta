from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training.component_context import ComponentContext
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment


@dataclass(slots=True)
class Loss:
    """Base class coordinating rollout and training behaviour for concrete losses."""

    policy: Policy
    trainer_cfg: Any
    env: TrainingEnvironment
    device: torch.device
    instance_name: str
    loss_cfg: Any

    policy_experience_spec: Composite | None = None
    replay: Experience | None = None
    loss_tracker: dict[str, list[float]] | None = None
    _zero_tensor: Tensor | None = None
    _context: ComponentContext | None = None

    rollout_start_epoch: int = 0
    rollout_end_epoch: float = float("inf")
    train_start_epoch: int = 0
    train_end_epoch: float = float("inf")
    rollout_cycle_length: int | None = None
    rollout_active_in_cycle: list[int] | None = None
    train_cycle_length: int | None = None
    train_active_in_cycle: list[int] | None = None

    def __post_init__(self) -> None:
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.loss_tracker = defaultdict(list)
        self._zero_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._configure_schedule()

    def attach_context(self, context: ComponentContext) -> None:
        """Register the shared trainer context for this loss instance."""
        self._context = context

    def _require_context(self, context: ComponentContext | None = None) -> ComponentContext:
        if context is not None:
            self._context = context
            return context
        if self._context is None:
            raise RuntimeError("Loss has not been attached to a ComponentContext")
        return self._context

    def get_experience_spec(self) -> Composite:
        """Optional extension of the experience replay buffer spec required by this loss."""
        return Composite()

    # --------- Control flow hooks; override in subclasses when custom behaviour is needed ---------

    def on_new_training_run(self, context: ComponentContext | None = None) -> None:
        """Called at the very beginning of a training epoch."""
        self._require_context(context)

    def on_rollout_start(self, context: ComponentContext | None = None) -> None:
        """Called before starting a rollout phase."""
        self._ensure_context(context)
        self.policy.reset_memory()

    def rollout(self, td: TensorDict, context: ComponentContext | None = None) -> None:
        """Rollout step executed while experience buffer requests more data."""
        ctx = self._ensure_context(context)
        if not self._should_run("rollout", ctx.epoch):
            return
        if ctx.training_env_id is None:
            raise RuntimeError("ComponentContext.training_env_id must be set before calling Loss.rollout")
        self.run_rollout(td, ctx)

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Override in subclasses to implement rollout logic."""
        return

    def train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext | None,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Training step executed while scheduler allows it."""
        ctx = self._ensure_context(context)
        if not self._should_run("train", ctx.epoch):
            return self._zero(), shared_loss_data, False
        return self.run_train(shared_loss_data, ctx, mb_idx)

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Override in subclasses to implement training logic."""
        return self._zero(), shared_loss_data, False

    def on_mb_end(self, context: ComponentContext | None, mb_idx: int) -> None:
        """Hook executed at the end of each minibatch."""
        self._ensure_context(context)

    def on_train_phase_end(self, context: ComponentContext | None = None) -> None:
        """Hook executed after the training phase completes."""
        self._ensure_context(context)

    def save_loss_states(self, context: ComponentContext | None = None) -> None:
        """Save loss states at the end of training (optional)."""
        self._ensure_context(context)

    # Scheduling helpers
    def _should_run(self, phase: str, epoch: int) -> bool:
        start = getattr(self, f"{phase}_start_epoch")
        end = getattr(self, f"{phase}_end_epoch")
        if not (start <= epoch < end):
            return False

        cycle_length = getattr(self, f"{phase}_cycle_length")
        active = getattr(self, f"{phase}_active_in_cycle") or []
        if not cycle_length or not active:
            return True

        # Epoch is 0-indexed; schedule uses 1-indexed values
        epoch_in_cycle = (epoch % cycle_length) + 1
        return epoch_in_cycle in active

    # End scheduling helpers

    def _configure_schedule(self) -> None:
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
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.loss_tracker.items()}

    def zero_loss_tracker(self) -> None:
        """Zero all values in the loss tracker."""
        self.loss_tracker.clear()

    # Internal utilities -------------------------------------------------

    def _ensure_context(self, context: ComponentContext | None) -> ComponentContext:
        if context is not None:
            self._context = context
            return context
        if self._context is None:
            raise RuntimeError("Loss has not been attached to a ComponentContext")
        return self._context

    def _zero(self) -> Tensor:
        assert self._zero_tensor is not None
        return self._zero_tensor

    def attach_replay_buffer(self, experience: Experience) -> None:
        """Attach the replay buffer to the loss."""
        self.replay = experience

    # End utility helpers
