from typing import Any, ClassVar, List, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.cogworks.curriculum import CurriculumConfig, env_curriculum
from metta.rl.loss.loss_config import LossConfig
from metta.rl.training.heartbeat import HeartbeatConfig
from metta.rl.training.scheduler import HyperparameterSchedulerConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder.envs import make_arena
from mettagrid.config import Config


class OptimizerConfig(Config):
    type: Literal["adam", "muon"] = "adam"
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


class InitialPolicyConfig(Config):
    uri: str | None = None
    type: Literal["top", "latest", "specific"] = "top"
    range: int = Field(default=1, gt=0)
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(Config):
    checkpoint_interval: int = Field(default=30, ge=0)
    checkpoint_dir: str | None = Field(default=None)
    remote_prefix: str | None = Field(default=None)


class EvaluationConfig(Config):
    simulations: List[SimulationConfig] = Field(default_factory=list)
    replay_dir: str | None = Field(default=None)

    evaluate_interval: int = Field(default=50, ge=0)
    evaluate_remote: bool = Field(default=True)
    evaluate_local: bool = Field(default=True)
    skip_git_check: bool = Field(default=False)
    git_hash: str | None = Field(default=None)
    num_training_tasks: int = Field(default=1)


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

    curriculum: CurriculumConfig = env_curriculum(make_arena(num_agents=24))
    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: Optional[EvaluationConfig] = Field(default=EvaluationConfig())

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

        if self.evaluation and self.evaluation.evaluate_interval != 0:
            if self.evaluation.evaluate_interval < self.checkpoint.checkpoint_interval:
                raise ValueError(
                    "evaluate_interval must be at least as large as checkpoint_interval "
                    "to ensure policies are saved before evaluation"
                )
            if self.evaluation.evaluate_remote and not self.checkpoint.remote_prefix:
                # Without a remote prefix we cannot evaluate remotely; fall back to local evaluations only.
                self.evaluation.evaluate_remote = False

        return self
