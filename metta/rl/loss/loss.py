import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


def analyze_loss_alignment(
    shared_data: TensorDict,
    name1: str,
    name2: str,
    params: list[Tensor],
    tracker: dict[str, list[float]],
) -> None:
    """
    Computes alignment metrics between two losses stored in shared_data.

    Args:
        shared_data: TensorDict containing the unreduced loss vectors.
                     Expects keys '{name1}_loss_vec' and '{name2}_loss_vec'
                     to contain *attached* tensors (part of the graph).
        name1: Name of the first loss (e.g. "ks_val").
        name2: Name of the second loss (e.g. "ppo_val").
        params: List of policy parameters to compute gradients for.
        tracker: Dictionary list to append metrics to.
    """
    key1 = f"{name1}_loss_vec"
    key2 = f"{name2}_loss_vec"

    if key1 not in shared_data or key2 not in shared_data:
        return

    # 1. Retrieve attached loss vectors
    loss_vec1 = shared_data[key1]
    loss_vec2 = shared_data[key2]

    # Flatten for comparison
    vec1_flat = loss_vec1.flatten()
    vec2_flat = loss_vec2.flatten()

    if vec1_flat.shape != vec2_flat.shape:
        return

    # 2. Vector-level metrics (detached)
    with torch.no_grad():
        # Cosine similarity of loss vectors (batch alignment)
        loss_cos = F.cosine_similarity(vec1_flat, vec2_flat, dim=0)
        tracker[f"{name1}_{name2}_loss_cos"].append(float(loss_cos.item()))

        # Variance of loss difference
        loss_diff_var = (vec1_flat - vec2_flat).var()
        tracker[f"{name1}_{name2}_loss_diff_var"].append(float(loss_diff_var.item()))

    # 3. Gradient-level metrics
    params_with_grad = [p for p in params if p.requires_grad]
    if not params_with_grad:
        return

    # Compute gradients of the scalar means
    loss_scalar1 = loss_vec1.mean()
    loss_scalar2 = loss_vec2.mean()

    # We must retain_graph because the graph is needed for the actual optimization step later
    # allow_unused=True is needed because some policy parameters (e.g. actor head)
    # might not be part of the value loss graph.
    grads1 = torch.autograd.grad(
        loss_scalar1, params_with_grad, retain_graph=True, create_graph=False, allow_unused=True
    )

    grads2 = torch.autograd.grad(
        loss_scalar2, params_with_grad, retain_graph=True, create_graph=False, allow_unused=True
    )

    # Flatten and concatenate, treating None as zeros
    def flatten_grads(grads, params):
        flat_list = []
        for g, p in zip(grads, params, strict=True):
            if g is None:
                flat_list.append(torch.zeros_like(p).flatten())
            else:
                flat_list.append(g.flatten())
        return torch.cat(flat_list)

    grad1_flat = flatten_grads(grads1, params_with_grad).detach()
    grad2_flat = flatten_grads(grads2, params_with_grad).detach()

    # Cosine similarity of gradients
    grad_cos = F.cosine_similarity(grad1_flat, grad2_flat, dim=0)
    tracker[f"{name1}_{name2}_grad_cos"].append(float(grad_cos.item()))

    # Variance of gradient differences
    grad_diff_var = (grad1_flat - grad2_flat).var()
    tracker[f"{name1}_{name2}_grad_diff_var"].append(float(grad_diff_var.item()))


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


@dataclass(frozen=True)
class MaskMeta:
    agent_mask: Optional[torch.Tensor]
    mask_flat: Optional[torch.Tensor]
    mask_shape: tuple[int, ...]


@dataclass(slots=True)
class Loss:
    """Base class coordinating rollout and training behaviour for concrete losses."""

    policy: Policy
    trainer_cfg: "TrainerConfig"
    env: TrainingEnvironment
    device: torch.device
    instance_name: str
    cfg: LossConfig

    trainable_only: bool = False
    loss_profiles: set[int] | None = None
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

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        raise NotImplementedError("Losses must implement policy_output_keys")

    # --------- Control flow hooks; override in subclasses when custom behaviour is needed ---------

    def on_epoch_start(self, context: ComponentContext | None = None) -> None:
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

    def _training_env_id(self, context: ComponentContext, *, error: Optional[str] = None) -> slice:
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError(error or "ComponentContext.training_env_id is missing in rollout.")
        return env_slice

    def _zero(self) -> Tensor:
        assert self._zero_tensor is not None
        return self._zero_tensor

    def attach_replay_buffer(self, experience: Experience) -> None:
        """Attach the replay buffer to the loss."""
        self.replay = experience
        # Align with replay layout so slot metadata survives policy TD prep
        self.policy_experience_spec = experience.buffer.spec  # type: ignore[attr-defined]

    def _filter_minibatch(self, shared_loss_data: TensorDict) -> TensorDict:
        """Filter minibatch rows by slot profile/trainable flags.

        Returns a clone of ``shared_loss_data`` with mask metadata attached at
        ``_applied_mask`` so downstream consumers can apply the same masking to
        auxiliary tensors (e.g., advantages, priorities).
        """

        mb = shared_loss_data["sampled_mb"]

        mask = None
        if (profiles := getattr(self, "loss_profiles", None)) is not None:
            mask = torch.isin(mb["loss_profile_id"], torch.as_tensor(list(profiles), device=mb.device))
        if getattr(self, "trainable_only", False):
            mask = mb["is_trainable_agent"] if mask is None else mask & mb["is_trainable_agent"]
        if mask is None:
            return shared_loss_data

        mask_shape = mask.shape

        # If the agent mask is constant across env/time, keep the batch structure and
        # slice only the agent axis; otherwise fall back to flattened selection.
        mask_2d = mask.reshape(-1, mask_shape[-1])
        agent_mask = None
        if mask_2d.shape[0] > 0:
            first_row = mask_2d[0]
            if torch.equal(mask_2d, first_row.expand_as(mask_2d)):
                agent_mask = first_row

        filtered = shared_loss_data.clone()

        if agent_mask is not None:
            # Preserve leading batch dims, drop non-trainable agents.
            agent_idx = agent_mask.nonzero(as_tuple=False).squeeze(-1)

            def _mask_agent_tensor(t: torch.Tensor) -> torch.Tensor:
                # reshape to [-1, agents, ...], index-select agent axis, then reshape back
                lead = len(mask_shape)
                rest_shape = t.shape[lead:]
                t_view = t.reshape(-1, mask_shape[-1], *rest_shape)
                t_sel = t_view.index_select(1, agent_idx)
                new_shape = (*mask_shape[:-1], agent_idx.numel(), *rest_shape)
                return t_sel.reshape(new_shape)

            new_batch = (*mask_shape[:-1], int(agent_idx.numel()))
            filtered["sampled_mb"] = TensorDict(
                {k: _mask_agent_tensor(v) for k, v in mb.items()},
                batch_size=new_batch,
                device=mb.device,
            )
            mask_meta = MaskMeta(agent_mask=agent_mask, mask_flat=None, mask_shape=mask_shape)
            for key, value in list(filtered.items()):
                if key == "sampled_mb":
                    continue
                filtered[key] = self._apply_row_mask(value, mask_meta)

            filtered["_applied_mask"] = NonTensorData(mask_meta)
        else:
            # Mixed mask across batch: flatten and mask.
            mask_flat = mask.flatten()
            mb_flat = mb.flatten()

            filtered["sampled_mb"] = mb_flat[mask_flat]

            mask_meta = MaskMeta(agent_mask=None, mask_flat=mask_flat, mask_shape=mask_shape)
            for key, value in list(filtered.items()):
                if key == "sampled_mb":
                    continue
                filtered[key] = self._apply_row_mask(value, mask_meta)

            filtered["_applied_mask"] = NonTensorData(mask_meta)

        return filtered

    def _apply_row_mask(self, value: Any, mask_meta: MaskMeta) -> Any:
        """Apply either a flattened mask or per-agent mask to a value."""

        agent_mask = mask_meta.agent_mask
        mask_flat = mask_meta.mask_flat
        mask_shape = mask_meta.mask_shape

        def apply_flat(t: torch.Tensor) -> torch.Tensor:
            """Mask tensors when mask varies within the batch/agent grid."""

            assert mask_flat is not None
            target = mask_flat.numel()
            lead = len(mask_shape)

            if tuple(t.shape[:lead]) == mask_shape:
                return t.reshape(target, *t.shape[lead:])[mask_flat]

            if t.shape and t.shape[0] == target:
                return t[mask_flat]

            if lead > 1 and t.shape and t.shape[0] == mask_shape[0]:
                repeat = int(target // t.shape[0])
                expanded = t.repeat_interleave(repeat, dim=0)
                return expanded[mask_flat]

            return t

        def apply_agent(t: torch.Tensor) -> torch.Tensor:
            assert agent_mask is not None
            agent_idx = agent_mask.nonzero(as_tuple=False).squeeze(-1)
            lead = len(mask_shape)
            if t.dim() < lead:
                return t  # nothing to mask
            if tuple(t.shape[: lead - 1]) == tuple(mask_shape[:-1]) and t.shape[lead - 1] == mask_shape[-1]:
                return t.index_select(lead - 1, agent_idx)
            return t

        masker = apply_agent if agent_mask is not None else apply_flat

        if isinstance(value, NonTensorData):
            data = value.data
            return NonTensorData(masker(data)) if hasattr(data, "shape") else value
        if isinstance(value, torch.Tensor):
            return masker(value)
        if hasattr(value, "batch_size"):
            try:
                return masker(value)
            except Exception:
                return value
        return value

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
