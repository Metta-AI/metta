"""
Extended trainer configuration with environment factory support.

This module extends the existing TrainerConfig to support the new environment
factory system while maintaining backward compatibility.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import Field, model_validator

from metta.rl.env_factory import (
    AtariEnvFactoryConfig,
    EnvFactoryConfig,
    EnvironmentFactory,
    GymEnvFactoryConfig,
    MettaGridEnvFactoryConfig,
    ProcGenEnvFactoryConfig,
    create_environment_factory,
)
from metta.rl.trainer_config import TrainerConfig


class ExtendedTrainerConfig(TrainerConfig):
    """
    Extended trainer configuration with environment factory support.

    This extends the base TrainerConfig to support multiple environment types
    through the factory pattern while maintaining backward compatibility.
    """

    # Environment factory configuration (new)
    environment: Optional[EnvFactoryConfig] = Field(
        default=None,
        description="Environment factory configuration. If specified, takes precedence over curriculum/env.",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "ExtendedTrainerConfig":
        """Override parent validation to support environment factory config."""

        # First do the basic validation that parent does
        if self.minibatch_size > self.batch_size:
            raise ValueError("minibatch_size must be <= batch_size")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("batch_size must be divisible by minibatch_size")

        # Check that we have either environment factory config OR curriculum/env (backward compatibility)
        has_environment = self.environment is not None
        has_curriculum_or_env = self.curriculum is not None or self.env is not None

        if not has_environment and not has_curriculum_or_env:
            raise ValueError("Either 'environment' (factory config) or 'curriculum'/'env' (legacy) must be set")

        if has_environment and has_curriculum_or_env:
            # If both are specified, log a warning but prefer the new environment config
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Both 'environment' factory config and legacy 'curriculum'/'env' are specified. "
                "Using 'environment' config and ignoring legacy settings."
            )

        # Validate checkpoint intervals
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
        """Override to handle factory-based environments."""
        if self.environment is not None:
            # For factory-based environments, we need to extract the path
            if self.environment.env_type == "mettagrid":
                env_config = self.environment
                assert isinstance(env_config, MettaGridEnvFactoryConfig)
                return env_config.curriculum_path
            else:
                # For non-MettaGrid environments, return a placeholder
                return f"factory:{self.environment.env_type}"

        # Fall back to parent implementation for backward compatibility
        return super().curriculum_or_env

    def get_environment_factory(self) -> EnvironmentFactory:
        """Create an environment factory from the configuration."""
        if self.environment is not None:
            return create_environment_factory(self.environment)

        # Backward compatibility: create MettaGrid factory from curriculum/env
        curriculum_path = super().curriculum_or_env
        env_overrides = self.env_overrides

        config = MettaGridEnvFactoryConfig(curriculum_path=curriculum_path, env_overrides=env_overrides)

        return create_environment_factory(config)

    def is_mettagrid_environment(self) -> bool:
        """Check if this is a MettaGrid environment."""
        if self.environment is not None:
            return self.environment.env_type == "mettagrid"

        # Backward compatibility: assume MettaGrid if using curriculum/env
        return True

    def get_environment_type(self) -> str:
        """Get the environment type."""
        if self.environment is not None:
            return self.environment.env_type

        # Backward compatibility: assume MettaGrid
        return "mettagrid"


# Configuration builder functions for different environment types
def create_mettagrid_trainer_config(
    curriculum_path: str, env_overrides: Optional[Dict[str, Any]] = None, **trainer_kwargs: Any
) -> ExtendedTrainerConfig:
    """Create trainer config for MettaGrid environments."""

    environment_config = MettaGridEnvFactoryConfig(curriculum_path=curriculum_path, env_overrides=env_overrides or {})

    return ExtendedTrainerConfig(environment=environment_config, **trainer_kwargs)


def create_atari_trainer_config(
    game: str,
    frameskip: int = 4,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    max_episode_steps: Optional[int] = None,
    **trainer_kwargs: Any,
) -> ExtendedTrainerConfig:
    """Create trainer config for Atari environments."""

    environment_config = AtariEnvFactoryConfig(
        game=game,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_episode_steps=max_episode_steps,
    )

    # Atari-specific trainer defaults
    trainer_defaults = {
        "num_workers": 4,
        "batch_size": 32768,  # Smaller batch size for Atari
        "minibatch_size": 4096,
        "total_timesteps": 100_000_000,  # Typical for Atari
    }

    # Override defaults with provided kwargs
    trainer_defaults.update(trainer_kwargs)

    return ExtendedTrainerConfig(environment=environment_config, **trainer_defaults)


def create_gym_trainer_config(
    env_id: str, max_episode_steps: Optional[int] = None, render_mode: Optional[str] = None, **trainer_kwargs: Any
) -> ExtendedTrainerConfig:
    """Create trainer config for Gym environments."""

    environment_config = GymEnvFactoryConfig(
        env_id=env_id, max_episode_steps=max_episode_steps, render_mode=render_mode
    )

    # Gym-specific trainer defaults
    trainer_defaults = {
        "num_workers": 2,
        "batch_size": 16384,  # Smaller batch size for simple Gym envs
        "minibatch_size": 2048,
        "total_timesteps": 10_000_000,
    }

    trainer_defaults.update(trainer_kwargs)

    return ExtendedTrainerConfig(environment=environment_config, **trainer_defaults)


def create_procgen_trainer_config(
    env_name: str,
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: Literal["easy", "hard"] = "hard",
    **trainer_kwargs: Any,
) -> ExtendedTrainerConfig:
    """Create trainer config for ProcGen environments."""

    environment_config = ProcGenEnvFactoryConfig(
        env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode
    )

    # ProcGen-specific trainer defaults
    trainer_defaults = {
        "num_workers": 8,
        "batch_size": 65536,
        "minibatch_size": 8192,
        "total_timesteps": 200_000_000,  # ProcGen typically needs many timesteps
    }

    trainer_defaults.update(trainer_kwargs)

    return ExtendedTrainerConfig(environment=environment_config, **trainer_defaults)


# Utility functions for configuration loading
def load_extended_trainer_config_from_dict(config_dict: Dict[str, Any]) -> ExtendedTrainerConfig:
    """Load extended trainer config from dictionary."""
    return ExtendedTrainerConfig(**config_dict)


def create_extended_trainer_config_from_legacy(legacy_config: TrainerConfig) -> ExtendedTrainerConfig:
    """Convert legacy TrainerConfig to ExtendedTrainerConfig."""

    # Extract all fields from legacy config
    config_dict = legacy_config.model_dump()

    # Create MettaGrid environment config from legacy curriculum/env settings
    if legacy_config.curriculum or legacy_config.env:
        curriculum_path = legacy_config.curriculum_or_env
        environment_config = MettaGridEnvFactoryConfig(
            curriculum_path=curriculum_path, env_overrides=legacy_config.env_overrides
        )
        config_dict["environment"] = environment_config

    return ExtendedTrainerConfig(**config_dict)
