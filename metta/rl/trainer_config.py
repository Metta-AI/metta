from typing import Any, ClassVar, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.rl.hyperparameter_scheduler_config import HyperparameterSchedulerConfig
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
    # Interval at which to evaluate and generate replays: Type 2 arbitrary default
    evaluate_interval: int = Field(default=300, ge=0)  # 0 to disable
    replay_dir: str = Field(default="")
    evaluate_remote: bool = Field(default=False)
    skip_git_check: bool = Field(default=False)
    git_hash: str | None = Field(default=None)

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


class RepresentationLearningConfig(BaseModelWithForbidExtra):
    enabled: bool = False
    lambda_contrast: float = Field(default=0.1, ge=0)
    lambda_tc: float = Field(default=0.1, ge=0)
    lambda_pred: float = Field(default=0.1, ge=0)
    tau: float = Field(default=0.15, gt=0)
    alpha: float = Field(default=0.8, ge=0, le=1.0)
    num_negatives: int = Field(default=32, ge=0)
    steps_per_batch: int = Field(default=1, ge=0)
    loss_pred_type: Literal["cosine", "l2"] = "cosine"
    loss_tc_type: Literal["cosine", "l2"] = "cosine"


class TrainerConfig(BaseModelWithForbidExtra):
    # Core training parameters
    # Total timesteps: Type 2 arbitrary default
    total_timesteps: int = Field(default=50_000_000_000, gt=0)

    # PPO configuration
    ppo: PPOConfig = Field(default_factory=PPOConfig)

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

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

    # scheduler registry
    hyperparameter_scheduler: HyperparameterSchedulerConfig = Field(default_factory=HyperparameterSchedulerConfig)

    # Representation learning
    representation_learning: RepresentationLearningConfig = Field(default_factory=RepresentationLearningConfig)

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

        # it doesn't make sense to evaluate more often than we checkpoint since we need a saved policy to evaluate
        if (
            self.simulation.evaluate_interval != 0
            and self.simulation.evaluate_interval < self.checkpoint.checkpoint_interval
        ):
            raise ValueError(
                f"evaluate_interval must be at least as large as checkpoint_interval "
                f"({self.simulation.evaluate_interval} < {self.checkpoint.checkpoint_interval})"
            )
        if (
            self.simulation.evaluate_interval != 0
            and self.simulation.evaluate_interval < self.checkpoint.wandb_checkpoint_interval
        ):
            raise ValueError(
                f"evaluate_interval must be at least as large as wandb_checkpoint_interval "
                f"({self.simulation.evaluate_interval} < {self.checkpoint.wandb_checkpoint_interval})"
            )
        # Validate that we save policies locally at least as often as we upload to wandb
        if (
            self.checkpoint.wandb_checkpoint_interval != 0
            and self.checkpoint.checkpoint_interval != 0
            and self.checkpoint.wandb_checkpoint_interval < self.checkpoint.checkpoint_interval
        ):
            raise ValueError(
                f"wandb_checkpoint_interval must be at least as large as checkpoint_interval "
                f"to ensure policies exist locally before uploading to wandb "
                f"({self.checkpoint.wandb_checkpoint_interval} < {self.checkpoint.checkpoint_interval})"
            )

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

    trainer_cfg = cfg.trainer
    if not isinstance(trainer_cfg, DictConfig):
        raise ValueError("ListConfig is not supported")

    # Convert to dict and let OmegaConf handle all interpolations
    config_dict = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("trainer config must be a dict")

    # Some keys' defaults in TrainerConfig that are appropriate for multiprocessing but not serial
    # TODO: This should be handled via EnvConfig instead
    if cfg.get("vectorization") == "serial":
        config_dict["async_factor"] = 1
        config_dict["zero_copy"] = False

    # Set default paths if not provided
    if "checkpoint_dir" not in config_dict.setdefault("checkpoint", {}):
        config_dict["checkpoint"]["checkpoint_dir"] = f"{cfg.run_dir}/checkpoints"

    if "replay_dir" not in config_dict.setdefault("simulation", {}):
        config_dict["simulation"]["replay_dir"] = f"{cfg.run_dir}/replays/"

    if "profile_dir" not in config_dict.setdefault("profiler", {}):
        config_dict["profiler"]["profile_dir"] = f"{cfg.run_dir}/torch_traces"

    return TrainerConfig.model_validate(config_dict)
