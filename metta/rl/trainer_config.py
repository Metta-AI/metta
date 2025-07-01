from typing import Any, ClassVar, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.rl.kickstarter_config import KickstartConfig


class OptimizerConfig(BaseModelWithForbidExtra):
    type: Literal["adam", "muon"] = "adam"
    # Learning rate: Updated based on Joseph's sweep results
    learning_rate: float = Field(default=0.019, gt=0, le=1.0)
    # Beta1: Updated based on Joseph's sweep results
    beta1: float = Field(default=0.89, ge=0, le=1.0)
    # Beta2: Updated based on Joseph's sweep results
    beta2: float = Field(default=0.96, ge=0, le=1.0)
    # Epsilon: Updated based on Joseph's sweep results
    eps: float = Field(default=1.4e-7, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = Field(default=0, ge=0)


class LRSchedulerConfig(BaseModelWithForbidExtra):
    # LR scheduling disabled by default: Fixed LR often works well in RL
    enabled: bool = False
    # Annealing disabled: Common to use fixed LR for PPO
    anneal_lr: bool = False
    # No warmup by default: RL typically doesn't need warmup like supervised learning
    warmup_steps: int | None = None
    # Schedule type unset: Various options available when enabled
    schedule_type: Literal["linear", "cosine", "exponential"] | None = None


class PrioritizedExperienceReplayConfig(BaseModelWithForbidExtra):
    # Alpha: Updated based on Joseph's sweep results (0.79 enables strong prioritization)
    prio_alpha: float = Field(default=0.79, ge=0, le=1.0)
    # Beta0: Updated based on Joseph's sweep results
    prio_beta0: float = Field(default=0.59, ge=0, le=1.0)


class VTraceConfig(BaseModelWithForbidExtra):
    # V-trace rho clipping: Updated based on Joseph's sweep results (2.3 allows more off-policy correction)
    vtrace_rho_clip: float = Field(default=2.3, gt=0)
    # V-trace c clipping: Updated based on Joseph's sweep results (2.1 allows more off-policy bootstrapping)
    vtrace_c_clip: float = Field(default=2.1, gt=0)


class InitialPolicyConfig(BaseModelWithForbidExtra):
    uri: str | None = None
    # Type="top": Empirical best performing
    type: Literal["top", "latest", "specific"] = "top"
    # Range=1: Select single best policy, standard practice
    range: int = Field(default=1, gt=0)
    # Metric="epoch": Default sorting by training progress
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModelWithForbidExtra):
    # Checkpoint every 60s: Balance between recovery granularity and I/O overhead
    checkpoint_interval: int = Field(default=60, gt=0)
    # W&B every 5 min: Less frequent due to network overhead and storage costs
    wandb_checkpoint_interval: int = Field(default=300, ge=0)  # 0 to disable
    checkpoint_dir: str = Field(default="")

    @model_validator(mode="after")
    def validate_fields(self) -> "CheckpointConfig":
        assert self.checkpoint_dir, "checkpoint_dir must be set"
        return self


class SimulationConfig(BaseModelWithForbidExtra):
    # Evaluate interval: Type 2 arbitrary default
    evaluate_interval: int = Field(default=300, ge=0)  # 0 to disable
    # Replay interval: Type 2 arbitrary default
    replay_interval: int = Field(default=300, gt=0)
    replay_dir: str = Field(default="")

    @model_validator(mode="after")
    def validate_fields(self) -> "SimulationConfig":
        assert self.replay_dir, "replay_dir must be set"
        return self


class PPOConfig(BaseModelWithForbidExtra):
    # PPO hyperparameters
    # Clip coefficient: Updated based on Joseph's sweep results
    clip_coef: float = Field(default=0.15, gt=0, le=1.0)
    # Entropy coefficient: Updated based on Joseph's sweep results
    ent_coef: float = Field(default=0.017, ge=0)
    # GAE lambda: Updated based on Joseph's sweep results
    gae_lambda: float = Field(default=0.84, ge=0, le=1.0)
    # Gamma: Updated based on Joseph's sweep results (0.99 is standard)
    gamma: float = Field(default=0.99, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping: Updated based on Joseph's sweep results
    max_grad_norm: float = Field(default=2.6, gt=0)
    # Value function clipping: Updated based on Joseph's sweep results
    vf_clip_coef: float = Field(default=0.16, ge=0)
    # Value coefficient: Updated based on Joseph's sweep results
    vf_coef: float = Field(default=3.2, ge=0)
    # L2 regularization: Disabled by default, common in RL
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    # Advantage normalization: Standard PPO practice for stability
    norm_adv: bool = True
    # Value loss clipping: PPO best practice from implementation details
    clip_vloss: bool = True
    # Target KL: None allows unlimited updates, common for stable environments
    target_kl: float | None = None


class TrainerConfig(BaseModelWithForbidExtra):
    # Target for hydra instantiation
    target: str = Field(default="metta.rl.trainer.MettaTrainer", alias="_target_")

    # Core training parameters
    # Total timesteps: Type 2 arbitrary default
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
    # Zero copy: Performance optimization to avoid memory copies
    zero_copy: bool = True
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
    # CPU offload disabled: Keep tensors on GPU for speed
    cpu_offload: bool = False
    # Torch compile disabled by default for stability
    compile: bool = False
    # Reduce-overhead mode: Best for training loops when compile is enabled
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    # Profile every 10K epochs: Infrequent to minimize overhead
    profiler_interval_epochs: int = Field(default=10000, gt=0)

    # Distributed training
    # Forward minibatch: Type 2 default chosen arbitrarily
    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)
    # Async factor 2: Type 2 default chosen arbitrarily, overlaps computation and communication for efficiency
    async_factor: int = Field(default=2, gt=0)

    # Kickstart
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)

    # Base trainer fields
    # Number of parallel workers: No default, must be set based on hardware
    num_workers: int = Field(gt=0)
    env: str | None = None  # Environment config path
    # Default curriculum: Simple environment for initial experiments
    curriculum: str | None = "/env/mettagrid/curriculum/simple"
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)

    # Checkpoint configuration
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Simulation configuration
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # Grad mean variance logging
    # Disabled by default: Expensive diagnostic for debugging training instability
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
    if not isinstance(trainer_cfg, DictConfig):
        raise ValueError("ListConfig is not supported")

    if _target_ := trainer_cfg.get("_target_"):
        if _target_ != "metta.rl.trainer.MettaTrainer":
            raise ValueError(f"Unsupported trainer config: {_target_}")

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
