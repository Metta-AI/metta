import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment


@dataclass(slots=True)
class Loss:
    """Base class coordinating rollout and training behaviour for concrete losses.

    Args:
        policy (Policy): The policy being trained/evaluated by this loss.
        trainer_cfg (Any): Trainer configuration object (total timesteps, batch sizes, etc.).
        env (TrainingEnvironment): Vectorized training environment wrapper.
        device (torch.device): Device to place tensors and compute on.
        instance_name (str): Name used to identify this loss instance in logs/state.
        loss_cfg (Any): Concrete loss configuration.

    Optional attributes initialized at runtime:
        policy_experience_spec (Composite | None): Spec for experience fields required by the policy.
        replay (Experience | None): Attached replay buffer, when relevant.
        loss_tracker (dict[str, list[float]] | None): Aggregated per-epoch metrics.
        _zero_tensor (Tensor | None): Pre-allocated zero scalar on device for cheap returns.
        _context (ComponentContext | None): Shared trainer context, attached externally.
    """

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

    _state_attrs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.loss_tracker = defaultdict(list)
        self._zero_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.register_state_attr("loss_tracker")
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
        if not self._loss_gate_allows("rollout", ctx):
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
        if not self._loss_gate_allows("train", ctx):
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
    def _loss_gate_allows(self, phase: str, context: ComponentContext) -> bool:
        gates = getattr(context, "loss_run_gates", None)
        if not gates:
            return True
        entry = gates.get(self.instance_name) or gates.get(self.__class__.__name__.lower())
        if not entry:
            return True
        return bool(entry.get(phase, True))

    # End scheduling helpers

    def _configure_schedule(self) -> None:
        """No-op: scheduling is now controlled by trainer-level scheduler."""
        self._rollout_sched = None
        self._train_sched = None

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

    # ------------------------------------------------------------------
    # State dict helpers (mirrors torch.nn.Module semantics)
    # ------------------------------------------------------------------
    def register_state_attr(self, *names: str) -> None:
        """Register attributes that should be persisted in the loss state."""

        for name in names:
            if not hasattr(self, name):
                raise AttributeError(f"Loss has no attribute '{name}' to register for state tracking")
            self._state_attrs.add(name)

    def state_dict(self) -> OrderedDict[str, Any]:
        """Return a CPU-friendly snapshot of registered attributes."""

        state = OrderedDict()
        for name in sorted(self._state_attrs):
            value = getattr(self, name)
            state[name] = self._clone_state_value(value)
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool = True) -> tuple[list[str], list[str]]:
        """Restore registered attributes from a state dictionary."""

        missing_keys: list[str] = [name for name in self._state_attrs if name not in state_dict]
        unexpected_keys: list[str] = [name for name in state_dict.keys() if name not in self._state_attrs]

        for name in self._state_attrs - set(missing_keys):
            self._restore_state_value(name, state_dict[name])

        if strict and (missing_keys or unexpected_keys):
            missing_msg = f"Missing keys: {missing_keys}" if missing_keys else ""
            unexpected_msg = f"Unexpected keys: {unexpected_keys}" if unexpected_keys else ""
            separator = "; " if missing_msg and unexpected_msg else ""
            raise RuntimeError(f"Error loading loss state dict: {missing_msg}{separator}{unexpected_msg}")

        return missing_keys, unexpected_keys

    # ------------------------------------------------------------------
    # Internal helpers for state cloning/restoration
    # ------------------------------------------------------------------
    def _clone_state_value(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach().clone().cpu()
        if isinstance(value, Mapping):
            return {k: self._clone_state_value(v) for k, v in value.items()}
        if isinstance(value, defaultdict):
            return {k: copy.deepcopy(v) for k, v in value.items()}
        if hasattr(value, "clone") and callable(value.clone):
            return value.clone()
        return copy.deepcopy(value)

    def _restore_state_value(self, name: str, stored_value: Any) -> None:
        current = getattr(self, name, None)

        if isinstance(current, Tensor):
            tensor = stored_value if isinstance(stored_value, Tensor) else torch.as_tensor(stored_value)
            setattr(self, name, tensor.to(device=current.device, dtype=current.dtype))
            return

        if isinstance(current, defaultdict):
            rebuilt = defaultdict(current.default_factory)
            for key, value in (stored_value or {}).items():
                rebuilt[key] = copy.deepcopy(value)
            setattr(self, name, rebuilt)
            return

        if isinstance(current, dict):
            setattr(self, name, {k: copy.deepcopy(v) for k, v in (stored_value or {}).items()})
            return

        if isinstance(stored_value, Tensor):
            setattr(self, name, stored_value.to(device=self.device))
            return

        setattr(self, name, copy.deepcopy(stored_value))
