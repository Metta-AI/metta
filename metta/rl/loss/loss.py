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
        # Align the policy experience spec with the actual replay layout so slot metadata
        # (slot_id/loss_profile_id/is_trainable_agent) survives the policy TD prep step.
        if hasattr(experience, "buffer"):
            self.policy_experience_spec = experience.buffer.spec  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Shared filtering helpers (slot aware)
    # ------------------------------------------------------------------
    def _filter_minibatch(self, shared_loss_data: TensorDict) -> TensorDict:
        """Filter minibatch rows by loss_profile_id and trainable flag if present.

        Slot metadata (loss profile / trainable flags) can be recorded per timestep
        when ``bptt_horizon > 1``. We collapse those masks to the segment dimension
        so that row-aligned metadata (indices, priorities, advantages, etc.) keeps
        its 2-D layout instead of being flattened by a 2-D boolean mask.
        """

        if "sampled_mb" not in shared_loss_data.keys():
            return shared_loss_data

        mb = shared_loss_data["sampled_mb"]

        slot_mask = self._build_slot_mask(mb)
        if slot_mask is None:
            return shared_loss_data

        slot_mask = slot_mask.to(dtype=torch.bool, device=mb.device)
        row_mask = self._collapse_mask_to_segments(slot_mask, mb.batch_size)

        filtered = shared_loss_data.clone()
        filtered["sampled_mb"] = mb[row_mask]

        for key, value in list(filtered.items()):
            if key == "sampled_mb":
                continue
            masked_value = self._mask_row_aligned_value(value, row_mask, mb.batch_size)
            if masked_value is not None:
                filtered[key] = masked_value

        return filtered

    def _build_slot_mask(self, minibatch: TensorDict) -> torch.Tensor | None:
        """Construct a combined mask from loss_profile_id/trainable flags."""

        mask = None

        target_profiles = getattr(self, "loss_profiles", None)
        if target_profiles is not None and "loss_profile_id" in minibatch.keys():
            profile_ids = minibatch.get("loss_profile_id")
            if isinstance(profile_ids, torch.Tensor):
                allowed = torch.zeros_like(profile_ids, dtype=torch.bool)
                for pid in target_profiles:
                    allowed = allowed | (profile_ids == pid)
                mask = allowed if mask is None else mask & allowed

        if getattr(self, "trainable_only", False) and "is_trainable_agent" in minibatch.keys():
            trainable_mask = minibatch.get("is_trainable_agent")
            if isinstance(trainable_mask, torch.Tensor):
                mask = trainable_mask if mask is None else mask & trainable_mask

        return mask

    def _collapse_mask_to_segments(self, mask: torch.Tensor, batch_size: torch.Size) -> torch.Tensor:
        """Reduce an arbitrary slot mask to a 1-D segment mask.

        The minibatch layout is always ``[segments, bptt_horizon]``; row-aligned
        metadata only depends on the segment dimension. A 2-D mask produced from
        per-timestep metadata is broadcast to the minibatch shape and collapsed
        with ``any`` so that we retain the per-segment horizon structure.
        """

        if len(batch_size) == 0:
            raise ValueError("Cannot filter minibatch without batch dimensions")

        batch_ndim = len(batch_size)
        working_mask = mask

        if working_mask.dim() > batch_ndim:
            raise ValueError(
                f"Slot filter mask with shape {tuple(mask.shape)} has more dimensions than minibatch {tuple(batch_size)}"
            )

        if working_mask.dim() < batch_ndim:
            working_mask = working_mask.view(*working_mask.shape, *([1] * (batch_ndim - working_mask.dim())))

        try:
            working_mask = working_mask.expand(*batch_size)
        except RuntimeError as exc:
            raise ValueError(
                f"Slot filter mask with shape {tuple(mask.shape)} is not broadcastable to minibatch {tuple(batch_size)}"
            ) from exc

        return working_mask.reshape(batch_size[0], -1).any(dim=1)

    def _mask_row_aligned_value(
        self, value: Any, row_mask: torch.Tensor, batch_size: torch.Size
    ) -> Any | None:
        """Apply the segment mask to row-aligned metadata if shapes match."""

        segment_count = batch_size[0]

        if isinstance(value, NonTensorData):
            data = value.data
            if hasattr(data, "shape") and getattr(data, "shape", None):
                if data.shape[0] == segment_count:
                    mask = row_mask
                    if hasattr(data, "device") and mask.device != data.device:
                        mask = mask.to(device=data.device)
                    return NonTensorData(data[mask])
                return None
            try:
                bool_mask = row_mask.cpu().tolist()
                if len(data) == segment_count:
                    filtered_data = [entry for entry, keep in zip(data, bool_mask) if keep]
                    return NonTensorData(filtered_data)
            except TypeError:
                return None
            return None

        if isinstance(value, torch.Tensor):
            if value.shape[:1] == (segment_count,):
                return value[row_mask]
            return None

        if hasattr(value, "batch_size"):
            bs = value.batch_size
            if len(bs) >= 1 and bs[0] == segment_count:
                return value[row_mask]

        return None

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
