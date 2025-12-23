import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite

from metta.agent.policy import Policy
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from mettagrid.base_config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig


@dataclass(slots=True)
class MetricAccumulator:
    total: Tensor | float = 0.0
    count: int = 0

    def add(self, value: Tensor | float) -> None:
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() != 1:
                value = value.mean()
            if self.count == 0:
                self.total = value
            elif torch.is_tensor(self.total):
                self.total = self.total + value
            else:
                self.total = value + torch.tensor(float(self.total), device=value.device, dtype=value.dtype)
        else:
            if torch.is_tensor(self.total):
                self.total = self.total + torch.tensor(float(value), device=self.total.device, dtype=self.total.dtype)
            else:
                self.total = float(self.total) + float(value)
        self.count += 1

    def mean(self) -> Tensor | float:
        if self.count == 0:
            return 0.0
        if torch.is_tensor(self.total):
            return self.total / self.count
        return self.total / self.count


def _track_metric(tracker: dict[str, MetricAccumulator], key: str, value: Tensor | float) -> None:
    tracker[key].add(value)


def analyze_loss_alignment(
    shared_data: TensorDict,
    name1: str,
    name2: str,
    params: list[Tensor],
    tracker: dict[str, MetricAccumulator],
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
        _track_metric(tracker, f"{name1}_{name2}_loss_cos", loss_cos)

        # Variance of loss difference
        loss_diff_var = (vec1_flat - vec2_flat).var()
        _track_metric(tracker, f"{name1}_{name2}_loss_diff_var", loss_diff_var)

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
    _track_metric(tracker, f"{name1}_{name2}_grad_cos", grad_cos)

    # Variance of gradient differences
    grad_diff_var = (grad1_flat - grad2_flat).var()
    _track_metric(tracker, f"{name1}_{name2}_grad_diff_var", grad_diff_var)


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
    loss_tracker: dict[str, MetricAccumulator] | None = None
    _zero_tensor: Tensor | None = None
    _context: ComponentContext | None = None

    _state_attrs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.policy_experience_spec = self.policy.get_agent_experience_spec()
        self.loss_tracker = defaultdict(MetricAccumulator)
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
        stats: dict[str, float] = {}
        for key, acc in self.loss_tracker.items():
            mean = acc.mean()
            if torch.is_tensor(mean):
                stats[key] = float(mean.detach().cpu().item())
            else:
                stats[key] = float(mean)
        return stats

    def track_metric(self, key: str, value: Tensor | float) -> None:
        """Track a scalar metric without per-minibatch GPU syncs."""
        _track_metric(self.loss_tracker, key, value)

    def metric_mean(self, key: str) -> float:
        """Return a mean value for a tracked metric."""
        acc = self.loss_tracker.get(key)
        if acc is None:
            return 0.0
        mean = acc.mean()
        if torch.is_tensor(mean):
            return float(mean.detach().cpu().item())
        return float(mean)

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
