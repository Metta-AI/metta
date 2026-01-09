from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import Field, model_validator

from metta.rl.training.component import TrainerComponent
from metta.rl.training.scheduler import ScheduleRule
from mettagrid.base_config import Config


class SamplingConfig(Config):
    """Configuration for minibatch sampling during training."""

    method: Literal["sequential", "prioritized"] = "sequential"
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class TrajectoryIsolationLossRef(Config):
    """Reference a configured loss by key, with a stable unique name.

    Notes:
    - `loss` should match a key on `TrainerConfig.losses` (e.g. "ppo_actor").
    - `name` should be unique across the entire TrajectoryIsolationConfig so it can be
      used as a stable metric / logging namespace.
    """

    name: str = Field(min_length=1)
    loss: str = Field(min_length=1)


class TrajectoryIsolationSliceConfig(Config):
    """Declarative slice definition for trajectory isolation.

    Each slice corresponds to some fraction of the full environment workload and can
    be assigned one or more policies and zero or more losses.
    """

    name: str = Field(min_length=1)
    env_ratio: float = Field(gt=0.0, le=1.0)
    policies: list[str] = Field(min_length=1)
    losses: list[TrajectoryIsolationLossRef] = Field(default_factory=list)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @model_validator(mode="after")
    def _validate_fields(self) -> "TrajectoryIsolationSliceConfig":
        # policy refs should be unique (within the slice)
        if len(set(self.policies)) != len(self.policies):
            raise ValueError(f"Duplicate policy reference(s) in slice '{self.name}': {self.policies}")

        # loss names should be unique (within the slice)
        loss_names = [lr.name for lr in self.losses]
        if len(set(loss_names)) != len(loss_names):
            raise ValueError(f"Duplicate loss name(s) in slice '{self.name}': {loss_names}")

        return self


class TrajectoryIsolationConfig(Config):
    """Config for trajectory isolation slicing."""

    slices: list[TrajectoryIsolationSliceConfig] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.slices)

    @model_validator(mode="after")
    def _validate_fields(self) -> "TrajectoryIsolationConfig":
        slice_names = [s.name for s in self.slices]
        if len(set(slice_names)) != len(slice_names):
            raise ValueError(f"Duplicate slice name(s) in trajectory isolation config: {slice_names}")

        total_ratio = sum(float(s.env_ratio) for s in self.slices)
        if total_ratio > 1.0 + 1e-9:
            raise ValueError(f"Sum of slice env_ratio must be <= 1.0 (got {total_ratio})")

        # Loss names must be globally unique across slices.
        all_loss_names: list[str] = []
        for s in self.slices:
            all_loss_names.extend(lr.name for lr in s.losses)
        if len(set(all_loss_names)) != len(all_loss_names):
            raise ValueError(f"Duplicate loss name(s) across slices: {all_loss_names}")

        return self

    def validate_references(self, *, policy_assets: Mapping[str, Any], losses: Any) -> None:
        """Validate policy/loss references against available config objects.

        Args:
            policy_assets: mapping of policy-asset names (as used by TrainTool.policy_assets)
            losses: typically `TrainerConfig.losses` (LossesConfig)
        """

        available_policies = set(policy_assets.keys())
        available_losses = _available_loss_keys(losses)

        missing: list[str] = []
        for s in self.slices:
            missing_policies = [p for p in s.policies if p not in available_policies]
            if missing_policies:
                missing.append(
                    f"slice '{s.name}' references missing policy asset(s): {missing_policies} "
                    f"(available: {sorted(available_policies)})"
                )

            missing_losses = [lr for lr in s.losses if lr.loss not in available_losses]
            if missing_losses:
                missing.append(
                    f"slice '{s.name}' references missing loss key(s): {[lr.loss for lr in missing_losses]} "
                    f"(available: {sorted(available_losses)})"
                )

        if missing:
            raise ValueError("Invalid TrajectoryIsolationConfig references:\n- " + "\n- ".join(missing))


def _available_loss_keys(losses: Any) -> set[str]:
    """Return loss-config keys available on a LossesConfig-like object."""

    if losses is None:
        return set()

    # LossesConfig defines __iter__ yielding (name, cfg) for all loss configs.
    try:
        return {str(name) for name, _cfg in losses}
    except Exception:
        pass

    # Pydantic v2: model_fields on the class.
    fields = getattr(getattr(losses, "__class__", object), "model_fields", None)
    if isinstance(fields, dict):
        return {str(k) for k in fields.keys()}

    return set()


class TrajectoryIsolator(TrainerComponent):
    """Runtime controller for trajectory isolation."""

    def __init__(
        self,
        config: TrajectoryIsolationConfig | None = None,
        *,
        rules: list[ScheduleRule] | None = None,
    ) -> None:
        super().__init__(epoch_interval=1, step_interval=0)
        self.config = config or TrajectoryIsolationConfig()
        self.rules: list[ScheduleRule] = list(rules or [])

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        # Attach to context so other components can discover it.
        context.trajectory_isolator = self
        # Initialize scheduled values for epoch 0 / first rollout.
        self.apply_rules()

    def apply_rules(self) -> None:
        if not self.rules:
            return
        for rule in self.rules:
            rule.apply(obj=self.config, ctx=self.context)

    # ----------------- Trainer callbacks -----------------
    def on_rollout_end(self) -> None:
        # Allow step/metric-driven rules to update between rollout and train.
        self.apply_rules()

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        # Prepare values for next epoch's rollout.
        self.apply_rules()

    # ----------------- Future integration points -----------------
    def route_rollout(self, td: Any, *, training_env_id: slice, context: Any) -> None:
        """Hook called from the core training loop near inference.

        The implementation will eventually split `td` by slice, forward the appropriate policy/policies,
        and stitch outputs (including actions) back into `td`.
        """

        _ = (td, training_env_id, context)
        return

    def build_eval_plan(self, *args: Any, **kwargs: Any) -> None:
        """Hook called from the evaluator to plan per-slice evaluations.

        This will eventually return a plan describing per-slice policy specs and episode allocations.
        """

        _ = (args, kwargs)
        return None
