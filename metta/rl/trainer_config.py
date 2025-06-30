from typing import Any, ClassVar, Literal

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.rl.kickstarter_config import KickstartConfig


class OptimizerConfig(BaseModelWithForbidExtra):
    type: Literal["adam", "muon"] = "adam"
    learning_rate: float = Field(default=0.0004573146765703167, gt=0, le=1.0)
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    eps: float = Field(default=1e-12, gt=0)
    weight_decay: float = Field(default=0, ge=0)


class LRSchedulerConfig(BaseModelWithForbidExtra):
    enabled: bool = False
    anneal_lr: bool = False
    warmup_steps: int | None = None
    schedule_type: Literal["linear", "cosine", "exponential"] | None = None


class PrioritizedExperienceReplayConfig(BaseModelWithForbidExtra):
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(BaseModelWithForbidExtra):
    vtrace_rho_clip: float = Field(default=1.0, gt=0)
    vtrace_c_clip: float = Field(default=1.0, gt=0)


class InitialPolicyConfig(BaseModelWithForbidExtra):
    uri: str | None = None
    type: Literal["top", "latest", "specific"] = "top"
    range: int = Field(default=1, gt=0)
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModelWithForbidExtra):
    checkpoint_interval: int = Field(default=60, gt=0)
    wandb_checkpoint_interval: int = Field(default=300, ge=0)  # 0 to disable
    checkpoint_dir: str = Field(default="")

    @model_validator(mode="after")
    def validate_fields(self) -> "CheckpointConfig":
        assert self.checkpoint_dir, "checkpoint_dir must be set"
        return self


class SimulationConfig(BaseModelWithForbidExtra):
    evaluate_interval: int = Field(default=300, ge=0)  # 0 to disable
    replay_interval: int = Field(default=300, gt=0)
    replay_dir: str = Field(default="")

    @model_validator(mode="after")
    def validate_fields(self) -> "SimulationConfig":
        assert self.replay_dir, "replay_dir must be set"
        return self


class PPOConfig(BaseModelWithForbidExtra):
    # PPO hyperparameters
    clip_coef: float = Field(default=0.1, gt=0, le=1.0)
    ent_coef: float = Field(default=0.0021, ge=0)
    gae_lambda: float = Field(default=0.916, ge=0, le=1.0)
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    max_grad_norm: float = Field(default=0.5, gt=0)
    vf_clip_coef: float = Field(default=0.1, ge=0)
    vf_coef: float = Field(default=0.44, ge=0)
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None


class TrainerConfig(BaseModelWithForbidExtra):
    # Target for hydra instantiation
    target: str = Field(default="metta.rl.trainer.MettaTrainer", alias="_target_")

    # Core training parameters
    total_timesteps: int = Field(default=50_000_000_000, gt=0)

    # PPO configuration
    ppo: PPOConfig = Field(default_factory=PPOConfig)

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)

    # Experience replay
    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

    # V-trace
    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    # System configuration
    zero_copy: bool = True
    require_contiguous_env_ids: bool = False
    verbose: bool = True

    # Batch configuration
    batch_size: int = Field(default=524288, gt=0)
    minibatch_size: int = Field(default=16384, gt=0)
    bptt_horizon: int = Field(default=64, gt=0)
    update_epochs: int = Field(default=1, gt=0)
    scale_batches_by_world_size: bool = False

    # Performance configuration
    cpu_offload: bool = False
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    profiler_interval_epochs: int = Field(default=10000, gt=0)

    # Distributed training
    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)
    async_factor: int = Field(default=2, gt=0)

    # Kickstart
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)

    # Base trainer fields
    num_workers: int = Field(gt=0)
    env: str | None = None  # Environment config path
    curriculum: str | None = "/env/mettagrid/curriculum/simple"
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)

    # Checkpoint configuration
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Simulation configuration
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # Grad mean variance logging
    grad_mean_variance_interval: int = Field(default=0, ge=0)  # 0 to disable

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


def parse_trainer_config(
    cfg: DictConfig,
) -> TrainerConfig:
    """Parse trainer config from Hydra config.

    Args:
        cfg: The complete Hydra config (must contain trainer, run, and run_dir)
    """
    for key in ["trainer", "run", "run_dir"]:
        if not hasattr(cfg, key) or cfg[key] is None:
            raise ValueError(f"cfg must have a '{key}' field")

    trainer_cfg = cfg.trainer
    if isinstance(trainer_cfg, ListConfig):
        raise ValueError("ListConfig is not supported")

    if trainer_cfg._target_ != "metta.rl.trainer.MettaTrainer":
        raise ValueError(f"Unsupported trainer config: {trainer_cfg._target_}")

    # Convert to dict and let OmegaConf handle all interpolations
    config_dict = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("trainer config must be a dict")

    # Set default paths if not provided
    if "checkpoint_dir" not in config_dict.setdefault("checkpoint", {}):
        config_dict["checkpoint"]["checkpoint_dir"] = f"{cfg.run_dir}/checkpoints"

    if "replay_dir" not in config_dict.setdefault("simulation", {}):
        config_dict["simulation"]["replay_dir"] = f"s3://softmax-public/replays/{cfg.run}"

    return TrainerConfig.model_validate(config_dict)
