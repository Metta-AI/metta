import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


class LossConfig(Config):
    enabled: bool = Field(default=True)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "Loss":
        raise NotImplementedError("Subclasses must implement create method")


@dataclass(slots=True)
class Loss:
    """Base class coordinating rollout and training behaviour for concrete losses."""

    policy: Policy
    trainer_cfg: "TrainerConfig"
    env: TrainingEnvironment
    device: torch.device
    instance_name: str
    cfg: LossConfig

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
        filtered = self._filter_minibatch(shared_loss_data)
        return self.run_train(filtered, ctx, mb_idx)

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
        # Align with replay layout so slot metadata survives policy TD prep
        self.policy_experience_spec = experience.buffer.spec  # type: ignore[attr-defined]

    def _filter_minibatch(self, shared_loss_data: TensorDict) -> TensorDict:
        """Filter minibatch rows by slot profile/trainable flags."""

        mb = shared_loss_data["sampled_mb"]

        profiles = getattr(self, "loss_profiles", None)
        mask = None
        if profiles is not None:
            pid = mb["loss_profile_id"]
            mask = torch.isin(pid, torch.as_tensor(list(profiles), device=pid.device))

        if getattr(self, "trainable_only", False):
            train_mask = mb["is_trainable_agent"]
            mask = train_mask if mask is None else mask & train_mask

        if mask is None:
            return shared_loss_data

        row_mask = mask if mask.dim() == 1 else mask.any(dim=-1)

        filtered = shared_loss_data.clone()
        filtered["sampled_mb"] = mb[row_mask]

        for key, value in list(filtered.items()):
            if key == "sampled_mb":
                continue
            filtered[key] = self._apply_row_mask(value, row_mask)

        return filtered

    def _apply_row_mask(self, value: Any, row_mask: torch.Tensor) -> Any:
        rows = row_mask.numel()

        if isinstance(value, NonTensorData):
            data = value.data
            if hasattr(data, "shape") and getattr(data, "shape", None):
                if data.shape[0] != rows:
                    raise ValueError(f"Row-aligned NonTensorData expected leading dim {rows}, got {data.shape[0]}")
                mask = row_mask.to(device=getattr(data, "device", row_mask.device))
                return NonTensorData(data[mask])
            bool_mask = row_mask.cpu().tolist()
            if len(data) != rows:
                raise ValueError(f"Row-aligned sequence expected length {rows}, got {len(data)}")
            return NonTensorData([entry for entry, keep in zip(data, bool_mask, strict=False) if keep])

        if isinstance(value, torch.Tensor):
            if value.shape[:1] != (rows,):
                raise ValueError(f"Row-aligned tensor expected leading dim {rows}, got {value.shape[0]}")
            return value[row_mask]

        if hasattr(value, "batch_size"):
            bs = value.batch_size
            if len(bs) < 1 or bs[0] != rows:
                raise ValueError(f"Row-aligned object expected batch[0]=={rows}, got {bs}")
            return value[row_mask]

        raise TypeError(f"Unsupported row-aligned value type: {type(value).__name__}")

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
