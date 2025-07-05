import multiprocessing
from typing import Any, ClassVar, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.rl.kickstarter_config import KickstartConfig


class OptimizerConfig(BaseModelWithForbidExtra):
    type: Literal["adam", "muon"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = Field(default=0.0004573146765703167, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=1e-12, gt=0)
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
    # Alpha=0 disables prioritization (uniform sampling), Type 2 default to be updated by sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta0=0.6: From Schaul et al. (2016) "Prioritized Experience Replay" paper
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(BaseModelWithForbidExtra):
    # V-trace rho clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_rho_clip: float = Field(default=1.0, gt=0)
    # V-trace c clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_c_clip: float = Field(default=1.0, gt=0)


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
    replay_interval: int = Field(default=300, ge=0)  # 0 to disable
    replay_dir: str = Field(default="")

    @model_validator(mode="after")
    def validate_fields(self) -> "SimulationConfig":
        assert self.replay_dir, "replay_dir must be set"
        return self


class PPOConfig(BaseModelWithForbidExtra):
    # PPO hyperparameters
    # Clip coefficient: 0.1 is conservative, common range 0.1-0.3 from PPO paper (Schulman et al., 2017)
    clip_coef: float = Field(default=0.1, gt=0, le=1.0)
    # Entropy coefficient: Type 2 default chosen from sweep
    ent_coef: float = Field(default=0.0021, ge=0)
    # GAE lambda: Type 2 default chosen from sweep, deviates from typical 0.95, bias/variance tradeoff
    gae_lambda: float = Field(default=0.916, ge=0, le=1.0)
    # Gamma: Type 2 default chosen from sweep, deviates from typical 0.99, suggests shorter
    # effective horizon for multi-agent
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping: 0.5 is standard PPO default to prevent instability
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value function clipping: Matches policy clip for consistency
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value coefficient: Type 2 default chosen from sweep, balances policy vs value loss
    vf_coef: float = Field(default=0.44, ge=0)
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


class TorchProfilerConfig(BaseModelWithForbidExtra):
    interval_epochs: int = Field(default=10000, ge=0)  # 0 to disable
    # Upload location: None disables uploads, supports s3:// or local paths
    profile_dir: str = Field(default="")

    @property
    def enabled(self) -> bool:
        return self.interval_epochs > 0

    @model_validator(mode="after")
    def validate_fields(self) -> "TorchProfilerConfig":
        assert self.profile_dir, "profile_dir must be set"
        return self


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
    # Zero copy: Performance optimization to avoid memory copies (default assumes multiprocessing)
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
    profiler: TorchProfilerConfig = Field(default_factory=TorchProfilerConfig)

    # Distributed training
    # Forward minibatch: Type 2 default chosen arbitrarily
    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)
    # Async factor 2: Type 2 default chosen arbitrarily, overlaps computation and communication for efficiency
    #   (default assumes multiprocessing)
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


def create_trainer_config(
    cfg: DictConfig,
) -> TrainerConfig:
    """Create trainer config from Hydra config.

    Args:
        cfg: The complete Hydra config (must contain trainer, run, and run_dir)
    """
    for key in ["trainer", "run", "run_dir"]:
        if not hasattr(cfg, key) or cfg[key] is None:
            raise ValueError(f"cfg must have a '{key}' field")

    is_serial = cfg.get("vectorization") == "serial" if not OmegaConf.is_missing(cfg, "vectorization") else False

    trainer_cfg = cfg.trainer
    if not isinstance(trainer_cfg, DictConfig):
        raise ValueError("ListConfig is not supported")

    if _target_ := trainer_cfg.get("_target_"):
        if _target_ != "metta.rl.trainer.MettaTrainer":
            raise ValueError(f"Unsupported trainer config: {_target_}")

    async_factor_missing = OmegaConf.is_missing(trainer_cfg, "async_factor")
    zero_copy_missing = OmegaConf.is_missing(trainer_cfg, "zero_copy")
    num_workers_missing = OmegaConf.is_missing(trainer_cfg, "num_workers")

    # Convert to dict and let OmegaConf handle all interpolations
    config_dict = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("trainer config must be a dict")

    # Handle missing values
    if async_factor_missing:
        if is_serial:
            config_dict["async_factor"] = 1
        else:
            # Delete the key to use TrainerConfig's default
            config_dict.pop("async_factor", None)

    if zero_copy_missing:
        if is_serial:
            config_dict["zero_copy"] = False
        else:
            # Delete the key to use TrainerConfig's default
            config_dict.pop("zero_copy", None)

    if num_workers_missing:
        config_dict["num_workers"] = _calculate_default_num_workers(is_serial)

    # Set default paths if not provided
    if "checkpoint_dir" not in config_dict.setdefault("checkpoint", {}):
        config_dict["checkpoint"]["checkpoint_dir"] = f"{cfg.run_dir}/checkpoints"

    if "replay_dir" not in config_dict.setdefault("simulation", {}):
        config_dict["simulation"]["replay_dir"] = f"s3://softmax-public/replays/{cfg.run}"

    if "profile_dir" not in config_dict.setdefault("profiler", {}):
        config_dict["profiler"]["profile_dir"] = f"{cfg.run_dir}/torch_traces"

    return TrainerConfig.model_validate(config_dict)


def _calculate_default_num_workers(is_serial: bool) -> int:
    if is_serial:
        return 1

    # Use power of 2 for better batch size compatibility
    cpu_count = multiprocessing.cpu_count() or 1
    ideal_workers = cpu_count // 2

    # Round down to nearest power of 2
    num_workers = 1
    while num_workers * 2 <= ideal_workers:
        num_workers *= 2

    return max(1, num_workers)
