"""Simplified configuration system for Metta training."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adam"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class TrainerConfig:
    """Trainer configuration."""

    # Basic training params
    total_timesteps: int = 10_000_000
    batch_size: int = 32768
    minibatch_size: int = 2048
    num_workers: int = 1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 60
    wandb_checkpoint_interval: int = 300

    # Evaluation
    evaluate_interval: int = 300
    replay_interval: int = 300

    # Curriculum
    curriculum: str = "mettagrid/curriculum/simple"
    env_overrides: Dict[str, Any] = field(default_factory=lambda: {"desync_episodes": True})

    # Policy
    initial_policy_uri: Optional[str] = None

    # Hardware
    device: str = "cuda"
    compile: bool = False
    compile_mode: str = "default"

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Advanced
    forward_pass_minibatch_target_size: int = 1024
    bptt_horizon: int = 16
    clip_param: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Average reward
    average_reward: bool = False
    average_reward_alpha: float = 0.01

    # RNN
    use_rnn: bool = True

    # Misc
    cpu_offload: bool = True
    require_contiguous_env_ids: bool = False
    async_factor: int = 3
    profiler_interval_epochs: int = 0


@dataclass
class AgentConfig:
    """Agent configuration."""

    name: str = "simple_cnn"
    hidden_size: int = 128
    lstm_layers: int = 2
    clip_range: float = 0.0
    analyze_weights_interval: int = 300
    l2_norm_coeff: float = 0.0
    l2_init_coeff: float = 0.0
    l2_init_weight_update_interval: int = 0

    # Agent-specific params
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "metta"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Run identification
    run_name: str = "metta_run"
    run_dir: str = "./runs/metta_run"
    data_dir: str = "./data"

    # Components
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Environment
    vectorization: int = 1024

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        if "trainer" in data and isinstance(data["trainer"], dict):
            if "optimizer" in data["trainer"] and isinstance(data["trainer"]["optimizer"], dict):
                data["trainer"]["optimizer"] = OptimizerConfig(**data["trainer"]["optimizer"])
            data["trainer"] = TrainerConfig(**data["trainer"])

        if "agent" in data and isinstance(data["agent"], dict):
            data["agent"] = AgentConfig(**data["agent"])

        if "wandb" in data and isinstance(data["wandb"], dict):
            data["wandb"] = WandbConfig(**data["wandb"])

        return cls(**data)
