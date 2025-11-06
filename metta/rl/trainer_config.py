from __future__ import annotations

import typing

import pydantic

import metta.rl.training.heartbeat as training_heartbeat
import metta.rl.training.scheduler as training_scheduler
import mettagrid.base_config


def _default_loss_config() -> "metta.rl.loss.loss_config.LossConfig":
    from metta.rl.loss import loss_config

    return loss_config.LossConfig()


class OptimizerConfig(mettagrid.base_config.Config):
    type: typing.Literal["adam", "muon", "adamw_schedulefree", "sgd_schedulefree"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = pydantic.Field(default=0.001153637, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = pydantic.Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = pydantic.Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = pydantic.Field(default=3.186531e-07, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = pydantic.Field(default=0, ge=0)
    # ScheduleFree-specific parameters
    momentum: float = pydantic.Field(default=0.9, ge=0, le=1.0)  # Beta parameter for ScheduleFree
    warmup_steps: int = pydantic.Field(default=0, ge=0)  # Number of warmup steps for ScheduleFree


class InitialPolicyConfig(mettagrid.base_config.Config):
    uri: str | None = None
    type: typing.Literal["top", "latest", "specific"] = "top"
    range: int = pydantic.Field(default=1, gt=0)
    metric: str = "epoch"
    filters: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class TorchProfilerConfig(mettagrid.base_config.Config):
    interval_epochs: int = pydantic.Field(default=0, ge=0)  # 0 to disable
    profile_dir: str | None = pydantic.Field(default=None)

    @property
    def enabled(self) -> bool:
        return self.interval_epochs > 0

    @pydantic.model_validator(mode="after")
    def validate_fields(self) -> "TorchProfilerConfig":
        if self.enabled:
            assert self.profile_dir, "profile_dir must be set"
        return self


class TrainerConfig(mettagrid.base_config.Config):
    total_timesteps: int = pydantic.Field(default=50_000_000_000, gt=0)
    losses: "metta.rl.loss.loss_config.LossConfig" = pydantic.Field(default_factory=_default_loss_config)
    optimizer: OptimizerConfig = pydantic.Field(default_factory=OptimizerConfig)

    require_contiguous_env_ids: bool = False
    verbose: bool = True

    batch_size: int = pydantic.Field(default=524288, gt=0)
    minibatch_size: int = pydantic.Field(default=16384, gt=0)
    bptt_horizon: int = pydantic.Field(default=64, gt=0)
    update_epochs: int = pydantic.Field(default=1, gt=0)
    scale_batches_by_world_size: bool = False

    compile: bool = False
    compile_mode: typing.Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    detect_anomaly: bool = pydantic.Field(default=False)

    hyperparameter_scheduler: training_scheduler.HyperparameterSchedulerConfig = pydantic.Field(
        default_factory=training_scheduler.HyperparameterSchedulerConfig
    )
    heartbeat: typing.Optional[training_heartbeat.HeartbeatConfig] = pydantic.Field(
        default_factory=training_heartbeat.HeartbeatConfig
    )

    initial_policy: InitialPolicyConfig = pydantic.Field(default_factory=InitialPolicyConfig)
    profiler: TorchProfilerConfig = pydantic.Field(default_factory=TorchProfilerConfig)

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @pydantic.model_validator(mode="after")
    def validate_fields(self) -> "TrainerConfig":
        if self.minibatch_size > self.batch_size:
            raise ValueError("minibatch_size must be <= batch_size")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        return self
