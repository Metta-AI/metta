from typing import Any, ClassVar, List, Literal

from pydantic import ConfigDict, Field, model_validator

from metta.cogworks.curriculum import CurriculumConfig, env_curriculum
from metta.common.util.config import Config
from metta.mettagrid.config.envs import make_arena
from metta.rl.hyperparameter_scheduler_config import HyperparameterSchedulerConfig
from metta.rl.kickstarter_config import KickstartConfig
from metta.sim.simulation_config import SimulationConfig


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


class PrioritizedExperienceReplayConfig(Config):
    # Alpha=0 disables prioritization (uniform sampling), Type 2 default to be updated by sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta0=0.6: From Schaul et al. (2016) "Prioritized Experience Replay" paper
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(Config):
    # V-trace rho clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_rho_clip: float = Field(default=1.0, gt=0)
    # V-trace c clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_c_clip: float = Field(default=1.0, gt=0)


class InitialPolicyConfig(Config):
    uri: str | None = None
    # Type="top": Empirical best performing
    type: Literal["top", "latest", "specific"] = "top"
    # Range=1: Select single best policy, standard practice
    range: int = Field(default=1, gt=0)
    # Metric="epoch": Default sorting by training progress
    metric: str = "epoch"
    filters: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(Config):
    # Checkpoint every 5 epochs
    checkpoint_interval: int = Field(default=5, ge=0)
    # W&B every 5 epochs
    wandb_checkpoint_interval: int = Field(default=5, ge=0)
    checkpoint_dir: str | None = Field(default=None)


class EvaluationConfig(Config):
    simulations: List[SimulationConfig] = Field(default_factory=list)
    replay_dir: str = Field(default="s3://softmax-public/replays/")

    # Interval at which to evaluate and generate replays: Type 2 arbitrary default
    evaluate_interval: int = Field(default=50, ge=0)  # 0 to disable
    evaluate_remote: bool = Field(default=True)
    evaluate_local: bool = Field(default=False)
    skip_git_check: bool = Field(default=False)
    git_hash: str | None = Field(default=None)
    num_training_tasks: int = Field(default=1)

    @model_validator(mode="after")
    def validate_fields(self) -> "EvaluationConfig":
        assert self.replay_dir, "replay_dir must be set"
        return self


class PPOConfig(Config):
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

    # Forward minibatch
    forward_pass_minibatch_target_size: int = Field(default=4096, gt=0)

    # Async factor 2: overlaps computation and communication for efficiency
    async_factor: int = Field(default=2, gt=0)

    # scheduler registry
    hyperparameter_scheduler: HyperparameterSchedulerConfig = Field(default_factory=HyperparameterSchedulerConfig)

    # Kickstart
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)

    # Base trainer fields
    # Number of parallel workers: No default, must be set based on hardware
    rollout_workers: int = Field(default=1, gt=0)

    # Default curriculum: Simple environment for initial experiments
    curriculum: CurriculumConfig = env_curriculum(make_arena(num_agents=24))
    initial_policy: InitialPolicyConfig = Field(default_factory=InitialPolicyConfig)

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Simulation configuration
    evaluation: EvaluationConfig | None = Field(default=EvaluationConfig())

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

        # it doesn't make sense to evaluate more often than we checkpoint since we need a saved policy to evaluate
        if self.evaluation and self.evaluation.evaluate_interval != 0:
            if self.evaluation.evaluate_interval < self.checkpoint.checkpoint_interval:
                raise ValueError(
                    f"evaluate_interval must be at least as large as checkpoint_interval "
                    f"({self.evaluation.evaluate_interval} < {self.checkpoint.checkpoint_interval})"
                )
            if self.evaluation.evaluate_interval < self.checkpoint.wandb_checkpoint_interval:
                raise ValueError(
                    f"evaluate_interval must be at least as large as wandb_checkpoint_interval "
                    f"({self.evaluation.evaluate_interval} < {self.checkpoint.wandb_checkpoint_interval})"
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
