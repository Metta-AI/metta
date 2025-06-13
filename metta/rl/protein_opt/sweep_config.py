"""Pydantic models for sweep configuration validation."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ParameterDistribution(BaseModel):
    """Configuration for a single parameter's search space."""

    min: float
    max: float
    mean: Optional[float] = None
    scale: Optional[float] = 1.0
    distribution: Literal["uniform", "log_normal", "int_uniform", "uniform_pow2"]

    @field_validator("max")
    def validate_max_greater_than_min(cls, v, info):
        if "min" in info.data and v <= info.data["min"]:
            raise ValueError(f"max ({v}) must be greater than min ({info.data['min']})")
        return v

    @field_validator("distribution")
    def validate_distribution_type(cls, v, info):
        if v == "int_uniform" and "min" in info.data:
            # Ensure min/max are integers for int_uniform
            if not isinstance(info.data["min"], int) or not isinstance(info.data.get("max"), int):
                raise ValueError("min and max must be integers for int_uniform distribution")
        return v


class SweepParameters(BaseModel):
    """Parameter search space definitions."""

    class Config:
        extra = "allow"  # Allow any parameter names like trainer.learning_rate

    def __init__(self, **data):
        # Validate that all values are ParameterDistribution objects
        validated_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                validated_data[key] = ParameterDistribution(**value)
            else:
                raise ValueError(f"Parameter {key} must be a dictionary with distribution config")
        super().__init__(**validated_data)


class TrainerOverrides(BaseModel):
    """Trainer configuration overrides."""

    total_timesteps: Optional[int] = Field(None, gt=0)
    evaluate_interval: Optional[int] = Field(None, gt=0)
    checkpoint_interval: Optional[int] = Field(None, gt=0)
    minibatch_size: Optional[int] = Field(None, gt=0)
    batch_size: Optional[int] = Field(None, gt=0)
    num_steps: Optional[int] = Field(None, gt=0)
    update_epochs: Optional[int] = Field(None, gt=0)
    num_workers: Optional[int] = Field(None, ge=0)

    class Config:
        extra = "allow"  # Allow additional trainer fields

    @model_validator(mode="after")
    def validate_batch_sizes(self):
        if self.batch_size and self.minibatch_size:
            if self.batch_size % self.minibatch_size != 0:
                raise ValueError(
                    f"batch_size ({self.batch_size}) must be divisible by minibatch_size ({self.minibatch_size})"
                )
        return self


class EnvOverrides(BaseModel):
    """Environment configuration overrides."""

    class Config:
        extra = "allow"  # Allow any environment configuration


class SweepSection(BaseModel):
    """The sweep section of the configuration."""

    parameters: Dict[str, ParameterDistribution]
    metric: str = "reward"
    goal: Literal["maximize", "minimize"] = "maximize"
    trainer: Optional[TrainerOverrides] = None
    env: Optional[EnvOverrides] = None

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v):
        if not v:
            raise ValueError("At least one parameter must be defined in sweep.parameters")

        # Convert dict values to ParameterDistribution objects
        validated = {}
        for key, value in v.items():
            if isinstance(value, dict):
                validated[key] = ParameterDistribution(**value)
            elif isinstance(value, ParameterDistribution):
                validated[key] = value
            else:
                raise ValueError(f"Parameter {key} must be a distribution configuration")
        return validated


class SweepConfig(BaseModel):
    """Root configuration for sweep files."""

    # These fields should NOT be in sweep configs anymore
    program: Optional[str] = Field(None, deprecated=True)
    method: Optional[str] = Field(None, deprecated=True)
    project: Optional[str] = Field(None, deprecated=True)
    entity: Optional[str] = Field(None, deprecated=True)
    command: Optional[List[str]] = Field(None, deprecated=True)
    rollout_count: Optional[int] = Field(None, deprecated=True)

    # Valid fields
    num_random_samples: Optional[int] = Field(10, ge=1)
    num_samples: Optional[int] = Field(1, ge=1)
    resume: Optional[bool] = True

    # The main sweep configuration
    sweep: Optional[SweepSection] = None
    parameters: Optional[Dict[str, ParameterDistribution]] = None
    metric: Optional[str] = None
    goal: Optional[Literal["maximize", "minimize"]] = None

    # Additional sections that might be present
    eval: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    test: Optional[str] = None

    class Config:
        extra = "forbid"  # Don't allow unknown fields at root level

    @model_validator(mode="after")
    def validate_structure(self):
        """Ensure proper structure and warn about deprecated fields."""

        # Warn about deprecated fields
        deprecated_fields = []
        if self.program is not None:
            deprecated_fields.append("program")
        if self.method is not None:
            deprecated_fields.append("method")
        if self.project is not None:
            deprecated_fields.append("project")
        if self.entity is not None:
            deprecated_fields.append("entity")
        if self.command is not None:
            deprecated_fields.append("command")
        if self.rollout_count is not None:
            deprecated_fields.append("rollout_count (use --rollout-count CLI arg)")

        if deprecated_fields:
            import warnings

            warnings.warn(
                f"Deprecated fields found and will be ignored: {', '.join(deprecated_fields)}",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle two possible structures:
        # 1. Everything under 'sweep' (preferred)
        # 2. Parameters at root level (legacy)

        if self.sweep is None and self.parameters is not None:
            # Legacy structure - create sweep section from root-level fields
            self.sweep = SweepSection(
                parameters=self.parameters, metric=self.metric or "reward", goal=self.goal or "maximize"
            )
            # Clear root-level fields
            self.parameters = None
            self.metric = None
            self.goal = None
        elif self.sweep is None and self.parameters is None:
            raise ValueError(
                "No parameters defined. Either 'sweep.parameters' or root-level 'parameters' must be specified."
            )

        return self

    @model_validator(mode="after")
    def warn_about_root_overrides(self):
        """Warn if trainer/env overrides are at root level."""

        # Check for common mistake: trainer/env at root level
        if hasattr(self, "__dict__"):
            root_keys = set(self.__dict__.keys())
            if "trainer" in root_keys and self.sweep and not self.sweep.trainer:
                import warnings

                warnings.warn(
                    "Found 'trainer' at root level - it should be under 'sweep.trainer' to take effect!",
                    UserWarning,
                    stacklevel=2,
                )
            if "env" in root_keys and self.sweep and not self.sweep.env:
                import warnings

                warnings.warn(
                    "Found 'env' at root level - it should be under 'sweep.env' to take effect!",
                    UserWarning,
                    stacklevel=2,
                )

        return self


def validate_sweep_config(config_dict: Dict[str, Any]) -> SweepConfig:
    """Validate a sweep configuration dictionary.

    Args:
        config_dict: The configuration dictionary to validate

    Returns:
        Validated SweepConfig object

    Raises:
        ValidationError: If the configuration is invalid
    """
    return SweepConfig(**config_dict)


def load_and_validate_sweep_config(config_path: str) -> SweepConfig:
    """Load and validate a sweep configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated SweepConfig object

    Raises:
        ValidationError: If the configuration is invalid
        FileNotFoundError: If the config file doesn't exist
    """
    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return validate_sweep_config(config_dict)


if __name__ == "__main__":
    """CLI interface for validating sweep configurations.

    Usage:
        python -m metta.rl.protein_opt.sweep_config configs/sweep/my_sweep.yaml
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate sweep configuration files")
    parser.add_argument("config_path", type=str, help="Path to sweep configuration YAML file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed validation info")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        print(f"Validating {config_path}...")
        config = load_and_validate_sweep_config(str(config_path))

        print("✅ Configuration is valid!")

        if args.verbose:
            print("\nConfiguration summary:")
            if config.sweep:
                print(f"  - Parameters: {len(config.sweep.parameters)}")
                for param_name in config.sweep.parameters:
                    param = config.sweep.parameters[param_name]
                    print(f"    • {param_name}: {param.distribution} [{param.min}, {param.max}]")
                print(f"  - Metric: {config.sweep.metric}")
                print(f"  - Goal: {config.sweep.goal}")
                if config.sweep.trainer:
                    print("  - Has trainer overrides")
                if config.sweep.env:
                    print("  - Has environment overrides")
            else:
                print("  - Using legacy root-level parameters structure")

        # Check for warnings
        if config.rollout_count is not None:
            print("\n⚠️  Warning: rollout_count should be specified via --rollout-count CLI argument")

        deprecated = []
        if config.program is not None:
            deprecated.append("program")
        if config.method is not None:
            deprecated.append("method")
        if config.project is not None:
            deprecated.append("project")
        if config.entity is not None:
            deprecated.append("entity")
        if config.command is not None:
            deprecated.append("command")

        if deprecated:
            print(f"\n⚠️  Warning: Found deprecated fields that will be ignored: {', '.join(deprecated)}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
