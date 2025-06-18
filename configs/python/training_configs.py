"""Training configurations in Python.

This replaces various trainer/*.yaml and other config files with
Python dataclasses that are easier to understand and modify.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adam"  # "adam" or "muon"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    # Loss coefficients
    clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    vf_clip_coef: float = 0.1

    # Advantage estimation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_vloss: bool = True

    # Gradient clipping
    max_grad_norm: float = 0.5

    # KL target (None for no early stopping)
    target_kl: Optional[float] = None

    # V-trace parameters
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0


@dataclass
class TrainingConfig:
    """Main training configuration."""

    # Run settings
    run_name: str = "experiment"
    device: str = "cuda"
    seed: int = 0

    # Training duration
    total_timesteps: int = 50_000_000

    # Batch settings
    num_envs: int = 128
    rollout_length: int = 128
    batch_size: int = 262144
    minibatch_size: int = 16384
    update_epochs: int = 4

    # Checkpointing
    checkpoint_interval: int = 100
    eval_interval: int = 500
    wandb_checkpoint_interval: int = 1000

    # Learning rate schedule
    use_lr_schedule: bool = False
    lr_warmup_steps: int = 0

    # Experience replay
    prioritized_replay_alpha: float = 0.0  # 0 = uniform sampling
    prioritized_replay_beta0: float = 0.6

    # Other
    cpu_offload: bool = False
    compile_policy: bool = False
    compile_mode: str = "reduce-overhead"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0


# Preset configurations
def small_scale_config() -> TrainingConfig:
    """Configuration for small-scale experiments."""
    return TrainingConfig(
        total_timesteps=10_000_000,
        num_envs=32,
        batch_size=32768,
        minibatch_size=2048,
        checkpoint_interval=50,
        eval_interval=100,
    )


def medium_scale_config() -> TrainingConfig:
    """Configuration for medium-scale training."""
    return TrainingConfig(
        total_timesteps=100_000_000,
        num_envs=128,
        batch_size=131072,
        minibatch_size=8192,
        checkpoint_interval=100,
        eval_interval=500,
    )


def large_scale_config() -> TrainingConfig:
    """Configuration for large-scale training."""
    return TrainingConfig(
        total_timesteps=1_000_000_000,
        num_envs=512,
        batch_size=524288,
        minibatch_size=32768,
        checkpoint_interval=250,
        eval_interval=1000,
        compile_policy=True,
    )


def debug_config() -> TrainingConfig:
    """Configuration for debugging."""
    return TrainingConfig(
        total_timesteps=10000,
        num_envs=4,
        rollout_length=32,
        batch_size=128,
        minibatch_size=32,
        update_epochs=1,
        checkpoint_interval=10,
        eval_interval=10,
    )


# PPO variants
def ppo_default() -> PPOConfig:
    """Default PPO configuration."""
    return PPOConfig()


def ppo_high_entropy() -> PPOConfig:
    """PPO with higher entropy for exploration."""
    return PPOConfig(
        ent_coef=0.05,
        clip_coef=0.2,
    )


def ppo_stable() -> PPOConfig:
    """More conservative PPO for stable training."""
    return PPOConfig(
        clip_coef=0.05,
        vf_clip_coef=0.05,
        max_grad_norm=0.25,
        target_kl=0.01,
    )


# Optimizer presets
def adam_default() -> OptimizerConfig:
    """Default Adam optimizer."""
    return OptimizerConfig(type="adam")


def adam_low_lr() -> OptimizerConfig:
    """Adam with lower learning rate."""
    return OptimizerConfig(
        type="adam",
        learning_rate=1e-4,
    )


def muon_optimizer() -> OptimizerConfig:
    """Muon optimizer configuration."""
    return OptimizerConfig(
        type="muon",
        learning_rate=2e-4,
        beta1=0.95,
        beta2=0.9999,
        eps=1e-13,
    )


# Hardware-specific configs
def gpu_1x_config() -> TrainingConfig:
    """Config for single GPU training."""
    config = medium_scale_config()
    config.device = "cuda:0"
    return config


def gpu_4x_config() -> TrainingConfig:
    """Config for 4 GPU training."""
    config = large_scale_config()
    config.num_envs = 512
    config.batch_size = 524288
    return config


def cpu_config() -> TrainingConfig:
    """Config for CPU training."""
    config = small_scale_config()
    config.device = "cpu"
    config.num_envs = 16
    config.compile_policy = False
    return config


# Example: Combine configs for a specific experiment
def navigation_experiment() -> Dict[str, Any]:
    """Complete configuration for a navigation experiment."""
    return {
        "training": medium_scale_config(),
        "ppo": ppo_default(),
        "optimizer": adam_default(),
        "agent": "simple_cnn_agent",  # Name of agent config function
        "env": "navigation_eval_suite",  # Name of env suite function
    }
