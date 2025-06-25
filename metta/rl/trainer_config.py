from typing import Any, Literal

from omegaconf import DictConfig, ListConfig
from pydantic import Field, model_validator

from metta.util.typed_config import BaseModelWithForbidExtra


class OptimizerConfig(BaseModelWithForbidExtra):
    type: Literal["adam", "muon"] = "adam"
    learning_rate: float = Field(gt=0, le=1.0)
    beta1: float = Field(ge=0, le=1.0, default=0.9)
    beta2: float = Field(ge=0, le=1.0, default=0.999)
    eps: float = Field(gt=0, default=1e-8)
    weight_decay: float = Field(ge=0, default=0.0)


class LRSchedulerConfig(BaseModelWithForbidExtra):
    enabled: bool = False
    anneal_lr: bool = False
    warmup_steps: int | None = None
    schedule_type: Literal["linear", "cosine", "exponential"] | None = None


class PrioritizedExperienceReplayConfig(BaseModelWithForbidExtra):
    prio_alpha: float = Field(ge=0, le=1.0, default=0.0)
    prio_beta0: float = Field(ge=0, le=1.0, default=0.6)


class VTraceConfig(BaseModelWithForbidExtra):
    vtrace_rho_clip: float = Field(gt=0, default=1.0)
    vtrace_c_clip: float = Field(gt=0, default=1.0)


class KickstartTeacherConfig(BaseModelWithForbidExtra):
    teacher_uri: str
    action_loss_coef: float = Field(ge=0, default=1.0)
    value_loss_coef: float = Field(ge=0, default=1.0)


class KickstartConfig(BaseModelWithForbidExtra):
    teacher_uri: str | None = None
    action_loss_coef: float = Field(ge=0, default=1.0)
    value_loss_coef: float = Field(ge=0, default=1.0)
    anneal_ratio: float = Field(ge=0, le=1.0, default=0.65)
    kickstart_steps: int = Field(gt=0, default=1_000_000_000)
    additional_teachers: list[KickstartTeacherConfig] | None = None


class InitialPolicyConfig(BaseModelWithForbidExtra):
    uri: str | None = None
    type: Literal["top", "latest", "specific"] = "top"
    range: int = Field(gt=0, default=1)
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModelWithForbidExtra):
    # Target for hydra instantiation
    target: str = Field(alias="_target_")

    # Core training parameters
    resume: bool = True
    use_e3b: bool = False
    total_timesteps: int = Field(gt=0)

    # PPO hyperparameters
    clip_coef: float = Field(gt=0, le=1.0)
    ent_coef: float = Field(ge=0)
    gae_lambda: float = Field(ge=0, le=1.0)
    gamma: float = Field(ge=0, le=1.0)

    # Optimizer and scheduler
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)

    # Training parameters
    max_grad_norm: float = Field(gt=0)
    vf_clip_coef: float = Field(ge=0, default=0.1)
    vf_coef: float = Field(ge=0)
    l2_reg_loss_coef: float = Field(ge=0, default=0)
    l2_init_loss_coef: float = Field(ge=0, default=0)

    # Experience replay
    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

    # V-trace
    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    # Normalization and clipping
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None

    # System configuration
    zero_copy: bool = True
    require_contiguous_env_ids: bool = False
    verbose: bool = True

    # Batch configuration
    batch_size: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    bptt_horizon: int = Field(gt=0)
    update_epochs: int = Field(gt=0)

    # Performance configuration
    cpu_offload: bool = False
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    profiler_interval_epochs: int = Field(gt=0, default=10000)

    # Distributed training
    forward_pass_minibatch_target_size: int = Field(gt=0)
    async_factor: int = Field(gt=0, default=2)

    # Kickstart
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)

    # Base trainer fields
    num_workers: int | None = None
    num_steps: int = Field(gt=0, default=32)  # Number of environment steps
    env: str | None = None  # Environment config path
    curriculum: str | None = None
    env_overrides: dict[str, Any] = Field(default_factory=lambda: {"desync_episodes": True})
    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)

    # Checkpoint and evaluation
    checkpoint_dir: str = "${run_dir}/checkpoints"
    evaluate_interval: int = Field(gt=0, default=300)
    checkpoint_interval: int = Field(gt=0, default=60)
    wandb_checkpoint_interval: int = Field(gt=0, default=300)
    replay_interval: int = Field(gt=0, default=300)
    replay_dir: str = "s3://softmax-public/replays/${run}"

    # Average reward settings
    average_reward: bool = False
    average_reward_alpha: float = Field(gt=0, default=0.01)

    model_config = dict(
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


class MettaTrainerConfig(TrainerConfig):
    target: str = Field(default="metta.rl.trainer.MettaTrainer", alias="_target_")


def parse_trainer_config(cfg: DictConfig | ListConfig) -> TrainerConfig:
    if isinstance(cfg, ListConfig):
        raise ValueError("ListConfig is not supported")

    if cfg._target_ == "metta.rl.trainer.MettaTrainer":
        return MettaTrainerConfig.model_validate(cfg)

    raise ValueError(f"Unsupported trainer config: {cfg._target_}")
