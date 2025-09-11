from typing import ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.mettagrid.config import Config
from metta.rl.hyperparameter_scheduler_config import HyperparameterSchedulerConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.training.heartbeat import HeartbeatConfig


class OptimizerConfig(Config):
    type: Literal["adam", "muon"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = Field(default=0.000457, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=1e-12, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = Field(default=0, ge=0)


class TorchProfilerConfig(Config):
    interval_epochs: int = Field(default=0, ge=0)  # 0 to disable
    # Upload location: None disables uploads, supports s3:// or local paths
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
    # Core training parameters
    # Total timesteps: Type 2 arbitrary default
    total_timesteps: int = Field(default=50_000_000_000, gt=0)

    # Losses
    losses: LossConfig = Field(default_factory=LossConfig)

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    # Contiguous env IDs not required: More flexible env management
    require_contiguous_env_ids: bool = False
    # Verbose logging for debugging and monitoring
    verbose: bool = True

    # Batch configuration
    # Batch size: Type 2 default chosen from sweep
    batch_size: int = Field(default=524288, gt=0)
    # Minibatch: Type 2 default chosen from sweep
    minibatch_size: int = Field(default=16384, gt=0)
    # BPTT horizon: Type 2 default chosen arbitrarily
    bptt_horizon: int = Field(default=64, gt=0)
    # Single epoch: Type 2 default chosen arbitrarily PPO typically uses 3-10, but 1 works with large batches
    update_epochs: int = Field(default=1, gt=0)
    # Fixed batch size across GPUs for consistent hyperparameters
    scale_batches_by_world_size: bool = False

    # Performance configuration
    # Torch compile disabled by default for stability
    compile: bool = False
    # Reduce-overhead mode: Best for training loops when compile is enabled
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    # scheduler registry
    hyperparameter_scheduler: HyperparameterSchedulerConfig = Field(default_factory=HyperparameterSchedulerConfig)
    heartbeat: Optional[HeartbeatConfig] = Field(default_factory=HeartbeatConfig)

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
