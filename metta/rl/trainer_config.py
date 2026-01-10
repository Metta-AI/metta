from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.rl.nodes import default_nodes, node_specs_by_key
from metta.rl.training import HeartbeatConfig
from metta.rl.training.update_epochs_tuner import UpdateEpochAutoTunerConfig
from mettagrid.base_config import Config


class OptimizerConfig(Config):
    type: Literal["adam", "muon", "adamw_schedulefree", "sgd_schedulefree"] = "adamw_schedulefree"
    # Learning rate tuned from CvC sweep winners (schedule-free AdamW)
    learning_rate: float = Field(default=0.00737503357231617, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon tuned from CvC sweep winners
    eps: float = Field(default=5.0833278919526e-07, gt=0)
    # Weight decay: modest L2 regularization for AdamW-style optimizers
    weight_decay: float = Field(default=0.01, ge=0)
    # ScheduleFree-specific parameters
    momentum: float = Field(default=0.9, ge=0, le=1.0)  # Beta parameter for ScheduleFree
    warmup_steps: int = Field(default=1000, ge=0)  # Number of warmup steps for ScheduleFree


class SamplingConfig(Config):
    """Configuration for minibatch sampling during training."""

    method: Literal["sequential", "prioritized"] = "sequential"
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class RewardCenteringConfig(Config):
    enabled: bool = True
    beta: float = Field(default=1e-3, gt=0, le=1.0)
    initial_reward_mean: float = 0.0


class AdvantageConfig(Config):
    vtrace_rho_clip: float = Field(default=1.0, gt=0)
    vtrace_c_clip: float = Field(default=1.0, gt=0)

    # Average-reward baseline: replace r with (r - r_bar) and update r_bar via EMA.
    reward_centering: RewardCenteringConfig = Field(default_factory=RewardCenteringConfig)

    gamma: float = Field(default=1.0, ge=0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1.0)


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
    total_timesteps: int = Field(default=10_000_000_000, gt=0)
    nodes: dict[str, Any] = Field(default_factory=default_nodes)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)

    require_contiguous_env_ids: bool = False
    verbose: bool = True

    batch_size: int = Field(default=2_097_152, gt=0)
    minibatch_size: int = Field(default=16384, gt=0)
    bptt_horizon: int = Field(default=256, gt=0)
    update_epochs: int = Field(default=1, gt=0)
    scale_batches_by_world_size: bool = False

    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    detect_anomaly: bool = Field(default=False)

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
        object.__setattr__(self, "nodes", _normalize_nodes(self.nodes))

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


def _normalize_nodes(nodes: dict[str, Any]) -> dict[str, Any]:
    specs = node_specs_by_key()
    normalized = dict(nodes)

    for key, spec in specs.items():
        if key not in normalized:
            normalized[key] = spec.config_cls(enabled=spec.default_enabled)

    for key, value in list(normalized.items()):
        spec = specs.get(key)
        if spec is None:
            raise ValueError(f"Unknown node config '{key}'")
        if isinstance(value, spec.config_cls):
            continue
        normalized[key] = spec.config_cls.model_validate(value)

    return normalized
