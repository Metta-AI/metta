from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.rl.binding_config import LossProfileConfig, PolicyBindingConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.training import HeartbeatConfig
from mettagrid.base_config import Config


class OptimizerConfig(Config):
    type: Literal["adam", "muon", "adamw_schedulefree", "sgd_schedulefree"] = "adamw_schedulefree"
    # Learning rate: tuned for ScheduleFree AdamW (scaled down ~20% from the legacy Adam default)
    learning_rate: float = Field(default=0.00092, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=3.186531e-07, gt=0)
    # Weight decay: modest L2 regularization for AdamW-style optimizers
    weight_decay: float = Field(default=0.01, ge=0)
    # ScheduleFree-specific parameters
    momentum: float = Field(default=0.9, ge=0, le=1.0)  # Beta parameter for ScheduleFree
    warmup_steps: int = Field(default=1000, ge=0)  # Number of warmup steps for ScheduleFree


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


class TrainerConfig(Config):
    total_timesteps: int = Field(default=50_000_000_000, gt=0)
    losses: LossesConfig = Field(default_factory=LossesConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    policy_bindings: list[PolicyBindingConfig] | None = Field(
        default=None, description="Optional list of policy bindings; defaults to a single trainer policy binding."
    )
    agent_binding_map: list[str] | None = Field(
        default=None,
        description="Optional mapping (length=num_agents) assigning each agent index to a policy binding id.",
    )
    loss_profiles: dict[str, LossProfileConfig] = Field(
        default_factory=dict, description="Optional loss profiles keyed by name."
    )

    require_contiguous_env_ids: bool = False
    verbose: bool = True

    batch_size: int = Field(default=524288, gt=0)
    minibatch_size: int = Field(default=16384, gt=0)
    bptt_horizon: int = Field(default=64, gt=0)
    update_epochs: int = Field(default=1, gt=0)
    scale_batches_by_world_size: bool = False

    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    detect_anomaly: bool = Field(default=False)

    heartbeat: Optional[HeartbeatConfig] = Field(default_factory=HeartbeatConfig)

    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)
    profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)

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
        return self
