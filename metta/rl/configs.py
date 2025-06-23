"""Structured configs for RL components."""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class RolloutConfig:
    """Configuration for rollout collection."""

    num_steps: Optional[int] = None  # If None, fills experience buffer
    device: str = "cuda"


@dataclass
class PPOConfig:
    """Configuration for PPO optimization."""

    # Core PPO parameters
    clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training parameters
    update_epochs: int = 4
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_clip_coef: float = 0.1
    target_kl: Optional[float] = None

    # Regularization
    l2_reg_loss_coef: float = 0.0
    l2_init_loss_coef: float = 0.0

    # V-trace
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0

    # Prioritized experience replay
    use_per: bool = False
    per_alpha: float = 0.0
    per_beta0: float = 0.6


@dataclass
class TrainerConfig:
    """Configuration for the main training loop."""

    # Training duration
    total_timesteps: int = 10_000_000

    # Batch sizes
    batch_size: int = 2048
    minibatch_size: int = 256
    forward_pass_minibatch_target_size: int = 4096

    # Environment setup
    num_workers: int = 1
    async_factor: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    evaluate_interval: int = 100
    replay_interval: int = 0
    wandb_checkpoint_interval: int = 500

    # Optimization
    learning_rate: float = 3e-4
    optimizer_type: str = "adam"  # "adam" or "muon"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Compilation
    compile: bool = False
    compile_mode: str = "default"

    # Experience buffer
    bptt_horizon: int = 16
    cpu_offload: bool = False

    # Misc
    seed: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None


@dataclass
class ExperienceConfig:
    """Configuration for experience buffer."""

    batch_size: int
    minibatch_size: int
    bptt_horizon: int = 16
    cpu_offload: bool = False
    device: str = "cuda"


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    sim_suite_config: dict = field(default_factory=dict)
    stats_dir: str = "/tmp/stats"
    replay_dir: Optional[str] = None
