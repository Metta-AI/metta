"""Structured configuration for training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from metta.rl.configs import PPOConfig, TrainerConfig


@dataclass
class AgentConfig:
    """Configuration for agent architecture."""

    name: str = "simple_cnn"
    hidden_size: int = 256
    num_layers: int = 2

    # Weight initialization
    l2_init_weight_update_interval: int = 0
    analyze_weights_interval: int = 0
    clip_range: float = 0.0

    # Architecture-specific params
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for environment setup."""

    name: str = "/env/mettagrid/mettagrid"
    width: int = 15
    height: int = 15
    max_steps: int = 500
    num_agents: int = 1

    # Environment-specific overrides
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = False
    stages: List[Dict[str, Any]] = field(default_factory=list)
    transition_steps: int = 100_000


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adam"  # "adam" or "muon"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    # Learning rate scheduling
    lr_schedule_enabled: bool = False
    lr_schedule_type: str = "cosine"  # "cosine", "linear", "exponential"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    project: str = "metta"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class HardwareConfig:
    """Configuration for hardware and distributed training."""

    device: str = "cuda"
    num_gpus: int = 1
    num_workers: int = 1

    # Distributed training
    distributed: bool = False
    backend: str = "nccl"  # "nccl", "gloo"

    # Memory optimization
    cpu_offload: bool = False
    gradient_checkpointing: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    enabled: bool = True
    interval: int = 100

    # Simulation suites to run
    suites: List[str] = field(default_factory=lambda: ["navigation", "cooperation"])
    episodes_per_suite: int = 10

    # Replay generation
    generate_replays: bool = True
    replay_interval: int = 500


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Core components
    agent: AgentConfig = field(default_factory=AgentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Optional components
    curriculum: Optional[CurriculumConfig] = None
    wandb: Optional[WandbConfig] = None
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: str = "default"
    seed: Optional[int] = None
    notes: str = ""

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    replay_dir: str = "./replays"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        # Handle nested configs
        if "agent" in config_dict:
            config_dict["agent"] = AgentConfig(**config_dict["agent"])
        if "environment" in config_dict:
            config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])
        if "trainer" in config_dict:
            config_dict["trainer"] = TrainerConfig(**config_dict["trainer"])
        if "ppo" in config_dict:
            config_dict["ppo"] = PPOConfig(**config_dict["ppo"])
        if "optimizer" in config_dict:
            config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
        if "curriculum" in config_dict and config_dict["curriculum"]:
            config_dict["curriculum"] = CurriculumConfig(**config_dict["curriculum"])
        if "wandb" in config_dict and config_dict["wandb"]:
            config_dict["wandb"] = WandbConfig(**config_dict["wandb"])
        if "hardware" in config_dict:
            config_dict["hardware"] = HardwareConfig(**config_dict["hardware"])
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "__dict__"):
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result


# Preset configurations
def small_fast_config() -> TrainingConfig:
    """Small, fast configuration for testing."""
    return TrainingConfig(
        agent=AgentConfig(name="simple_cnn", hidden_size=128),
        environment=EnvironmentConfig(width=10, height=10, max_steps=100),
        trainer=TrainerConfig(
            total_timesteps=10_000,
            batch_size=512,
            minibatch_size=64,
        ),
        ppo=PPOConfig(
            update_epochs=2,
            clip_coef=0.1,
        ),
        evaluation=EvaluationConfig(enabled=False),
    )


def medium_config() -> TrainingConfig:
    """Medium configuration for regular training."""
    return TrainingConfig(
        agent=AgentConfig(name="simple_cnn", hidden_size=256),
        environment=EnvironmentConfig(width=15, height=15, max_steps=500),
        trainer=TrainerConfig(
            total_timesteps=10_000_000,
            batch_size=2048,
            minibatch_size=256,
        ),
        ppo=PPOConfig(),
        wandb=WandbConfig(enabled=True),
    )


def large_distributed_config() -> TrainingConfig:
    """Large configuration for distributed training."""
    return TrainingConfig(
        agent=AgentConfig(name="large_cnn", hidden_size=512, num_layers=3),
        environment=EnvironmentConfig(width=25, height=25, max_steps=1000, num_agents=4),
        trainer=TrainerConfig(
            total_timesteps=100_000_000,
            batch_size=8192,
            minibatch_size=512,
            num_workers=4,
        ),
        ppo=PPOConfig(
            clip_coef=0.2,
            update_epochs=8,
        ),
        optimizer=OptimizerConfig(
            type="muon",
            learning_rate=1e-3,
            lr_schedule_enabled=True,
        ),
        hardware=HardwareConfig(
            num_gpus=8,
            distributed=True,
            gradient_checkpointing=True,
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=[
                {"width": 10, "height": 10, "max_steps": 200},
                {"width": 15, "height": 15, "max_steps": 500},
                {"width": 25, "height": 25, "max_steps": 1000},
            ],
            transition_steps=1_000_000,
        ),
        wandb=WandbConfig(enabled=True, tags=["distributed", "large"]),
    )
