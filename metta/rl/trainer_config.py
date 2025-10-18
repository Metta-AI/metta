from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.rl.loss import LossConfig
from metta.rl.training import HeartbeatConfig, HyperparameterSchedulerConfig
from mettagrid.base_config import Config


class OptimizerConfig(Config):
    type: Literal["adam", "muon", "adamw_schedulefree", "sgd_schedulefree"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = Field(default=0.001153637, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=3.186531e-07, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = Field(default=0, ge=0)
    # ScheduleFree-specific parameters
    momentum: float = Field(default=0.9, ge=0, le=1.0)  # Beta parameter for ScheduleFree
    warmup_steps: int = Field(default=0, ge=0)  # Number of warmup steps for ScheduleFree


class InitialPolicyConfig(Config):
    uri: str | None = None
    type: Literal["top", "latest", "specific"] = "top"
    range: int = Field(default=1, gt=0)
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class TorchProfilerConfig(Config):
    interval_epochs: int = Field(default=0, ge=0)  # 0 to disable
    profile_dir: str | None = Field(default=None)

    @property
    def enabled(self) -> bool:
        return self.interval_epochs > 0

    @model_validator(mode="after")
    def validate_fields(self) -> "TorchProfilerConfig":
        if self.enabled:
            assert self.profile_dir, "profile_dir must be set"
        return self


class UpdateEpochAutoTunerConfig(Config):
    """Configuration for automatically tuning update epochs."""

    enabled: bool = False
    min_update_epochs: int = Field(default=1, ge=1)
    max_update_epochs: int = Field(default=8, ge=1)
    step_size: int = Field(default=1, ge=1)
    evaluation_epochs: int = Field(default=3, ge=1)
    warmup_epochs: int = Field(default=2, ge=0)
    cooldown_epochs: int = Field(default=2, ge=0)
    min_relative_improvement: float = Field(default=0.05, ge=0.0)
    metrics_window: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "UpdateEpochAutoTunerConfig":
        if self.max_update_epochs < self.min_update_epochs:
            raise ValueError("max_update_epochs must be >= min_update_epochs")
        return self


class TrainerConfig(Config):
    total_timesteps: int = Field(default=50_000_000_000, gt=0)
    losses: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    require_contiguous_env_ids: bool = False
    verbose: bool = True

    batch_size: int = Field(default=524288, gt=0)
    minibatch_size: int = Field(default=16384, gt=0)
    bptt_horizon: int = Field(default=64, gt=0)
    update_epochs: int = Field(default=1, gt=0)
    scale_batches_by_world_size: bool = False

    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    hyperparameter_scheduler: HyperparameterSchedulerConfig = Field(default_factory=HyperparameterSchedulerConfig)
    heartbeat: Optional[HeartbeatConfig] = Field(default_factory=HeartbeatConfig)

    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)
    profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)
    update_epochs_autotune: UpdateEpochAutoTunerConfig = Field(default_factory=UpdateEpochAutoTunerConfig)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "TrainerConfig":
        if self.minibatch_size > self.batch_size:
            raise ValueError("minibatch_size must be <= batch_size")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")
        auto_cfg = self.update_epochs_autotune
        if auto_cfg.enabled:
            if self.update_epochs < auto_cfg.min_update_epochs or self.update_epochs > auto_cfg.max_update_epochs:
                raise ValueError(
                    "update_epochs must be within [min_update_epochs, max_update_epochs] when autotune is enabled"
                )
        return self
