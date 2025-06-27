from typing import Any, ClassVar, Literal

from omegaconf import DictConfig, ListConfig
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra


class OptimizerConfig(BaseModelWithForbidExtra):
    type: Literal["adam", "muon"] = "adam"
    learning_rate: float = Field(gt=0, le=1.0)
    beta1: float = Field(ge=0, le=1.0)
    beta2: float = Field(ge=0, le=1.0)
    eps: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class LRSchedulerConfig(BaseModelWithForbidExtra):
    enabled: bool = False
    anneal_lr: bool = False
    warmup_steps: int | None
    schedule_type: Literal["linear", "cosine", "exponential"] | None


class PrioritizedExperienceReplayConfig(BaseModelWithForbidExtra):
    prio_alpha: float = Field(ge=0, le=1.0)
    prio_beta0: float = Field(ge=0, le=1.0)


class VTraceConfig(BaseModelWithForbidExtra):
    vtrace_rho_clip: float = Field(gt=0)
    vtrace_c_clip: float = Field(gt=0)


class KickstartTeacherConfig(BaseModelWithForbidExtra):
    teacher_uri: str
    action_loss_coef: float = Field(ge=0)
    value_loss_coef: float = Field(ge=0)


class KickstartConfig(BaseModelWithForbidExtra):
    teacher_uri: str | None
    action_loss_coef: float = Field(ge=0)
    value_loss_coef: float = Field(ge=0)
    anneal_ratio: float = Field(ge=0, le=1.0)
    kickstart_steps: int = Field(gt=0)
    additional_teachers: list[KickstartTeacherConfig] | None = None


class InitialPolicyConfig(BaseModelWithForbidExtra):
    uri: str | None
    type: Literal["top", "latest", "specific"]
    range: int = Field(gt=0)
    metric: str = "epoch"
    filters: dict[str, Any]


class TrainerConfig(BaseModelWithForbidExtra):
    # Target for hydra instantiation
    target: str = Field(alias="_target_")

    # Core training parameters
    total_timesteps: int = Field(gt=0)

    # PPO hyperparameters
    clip_coef: float = Field(gt=0, le=1.0)
    ent_coef: float = Field(ge=0)
    gae_lambda: float = Field(ge=0, le=1.0)
    gamma: float = Field(ge=0, le=1.0)

    # Optimizer and scheduler
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig

    # Training parameters
    max_grad_norm: float = Field(gt=0)
    vf_clip_coef: float = Field(ge=0)
    vf_coef: float = Field(ge=0)
    l2_reg_loss_coef: float = Field(ge=0)
    l2_init_loss_coef: float = Field(ge=0)

    # Experience replay
    prioritized_experience_replay: PrioritizedExperienceReplayConfig

    # V-trace
    vtrace: VTraceConfig

    # Normalization and clipping
    norm_adv: bool
    clip_vloss: bool
    target_kl: float | None

    # System configuration
    zero_copy: bool
    require_contiguous_env_ids: bool
    verbose: bool

    # Batch configuration
    batch_size: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    bptt_horizon: int = Field(gt=0)
    update_epochs: int = Field(gt=0)

    # Performance configuration
    cpu_offload: bool
    compile: bool
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"]
    profiler_interval_epochs: int = Field(gt=0)

    # Distributed training
    forward_pass_minibatch_target_size: int = Field(gt=0)
    async_factor: int = Field(gt=0)

    # Kickstart
    kickstart: KickstartConfig

    # Base trainer fields
    num_workers: int = Field(gt=0)
    env: str | None = None  # Environment config path
    curriculum: str | None = None
    env_overrides: dict[str, Any]
    initial_policy: InitialPolicyConfig

    # Checkpoint and evaluation
    checkpoint_dir: str = "${run_dir}/checkpoints"
    evaluate_interval: int = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)
    wandb_checkpoint_interval: int = Field(ge=0)  # 0 to disable
    replay_interval: int = Field(gt=0)
    replay_dir: str = "s3://softmax-public/replays/${run}"
    grad_mean_variance_interval: int = Field(ge=0)  # 0 to disable

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

        if not self.curriculum and not self.env:
            raise ValueError("curriculum or env must be set")
        return self

    @property
    def curriculum_or_env(self) -> str:
        if self.curriculum:
            return self.curriculum
        if self.env:
            return self.env
        raise ValueError("curriculum or env must be set")


def parse_trainer_config(cfg: DictConfig | ListConfig) -> TrainerConfig:
    if isinstance(cfg, ListConfig):
        raise ValueError("ListConfig is not supported")

    if cfg._target_ == "metta.rl.trainer.MettaTrainer":
        return TrainerConfig.model_validate(cfg)

    raise ValueError(f"Unsupported trainer config: {cfg._target_}")
