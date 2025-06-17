"""
Configuration validation decorators and utilities for Hydra applications.

This module provides decorators to add runtime type validation to Hydra-based
applications, ensuring configs match their expected Pydantic schemas.
"""

import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from omegaconf import DictConfig, ListConfig
from pydantic import ValidationError

from metta.util.types.base_config import BaseConfig, config_registry

T = TypeVar("T", bound=BaseConfig)


def validate_config(config_class: Type[T]) -> Callable:
    """
    Decorator to validate entire Hydra config against a Pydantic model.

    The decorated function will receive a validated instance of config_class
    instead of the raw DictConfig.

    Usage:
        @hydra.main(version_base=None, config_path="../configs", config_name="train")
        @validate_config(TrainerConfig)
        def train(cfg: TrainerConfig):
            # cfg is now a validated TrainerConfig instance
            print(cfg.learning_rate)  # Full IDE support!
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(cfg: Any) -> Any:
            if isinstance(cfg, ListConfig):
                raise TypeError(
                    "validate_config decorator received ListConfig instead of DictConfig. "
                    "This usually means the config file contains a list at the root level."
                )

            try:
                # Convert OmegaConf to Pydantic model
                validated_cfg = config_class(cfg)
                # Call original function with validated config
                return func(validated_cfg)
            except ValidationError as e:
                print(f"Configuration validation failed for {config_class.__name__}:")
                print(e)
                raise
            except Exception as e:
                print(f"Unexpected error validating config with {config_class.__name__}:")
                print(e)
                raise

        return wrapper

    return decorator


def validate_subconfig(
    field_name: str, config_class: Type[T], optional: bool = False, in_place: bool = True
) -> Callable:
    """
    Decorator to validate a specific field in the Hydra config.

    Args:
        field_name: Dot-separated path to the field (e.g., "trainer.optimizer")
        config_class: Pydantic model to validate against
        optional: If True, missing field is not an error
        in_place: If True, replaces the field with validated dict in the original config

    Usage:
        @hydra.main(version_base=None, config_path="../configs", config_name="train")
        @validate_subconfig("simulation", SimulationConfig)
        @validate_subconfig("trainer.optimizer", OptimizerConfig, optional=True)
        def train(cfg: DictConfig):
            # cfg.simulation is validated
            # cfg.trainer.optimizer is validated if present
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(cfg: Any) -> Any:
            if isinstance(cfg, ListConfig):
                raise TypeError(
                    f"validate_subconfig decorator received ListConfig instead of DictConfig. "
                    f"Cannot access field '{field_name}' in a list."
                )

            # Navigate to the field
            field_parts = field_name.split(".")
            current = cfg

            # Check if field exists
            try:
                for i, part in enumerate(field_parts):
                    current = current[part]
                    # Check if we hit a ListConfig along the way
                    if isinstance(current, ListConfig) and i < len(field_parts) - 1:
                        raise TypeError(
                            f"Encountered ListConfig at '{'.'.join(field_parts[: i + 1])}' "
                            f"while navigating to '{field_name}'. Cannot access dict keys in a list."
                        )
            except (KeyError, AttributeError) as e:
                if optional:
                    return func(cfg)
                raise ValueError(f"Required config field '{field_name}' not found: {e}") from e

            try:
                # Validate the subconfig
                validated_subconfig = config_class(current)

                if in_place:
                    # Replace with validated version as dict to maintain OmegaConf structure
                    current = cfg
                    for part in field_parts[:-1]:
                        current = current[part]
                    current[field_parts[-1]] = validated_subconfig.model_dump()

                return func(cfg)
            except ValidationError as e:
                print(f"Validation failed for config field '{field_name}':")
                print(e)
                raise

        return wrapper

    return decorator


def validate_configs(*validations: tuple[str, Type[BaseConfig], bool]) -> Callable:
    """
    Decorator to validate multiple config fields at once.

    Args:
        *validations: Tuples of (field_name, config_class, optional)

    Usage:
        @hydra.main(version_base=None, config_path="../configs", config_name="train")
        @validate_configs(
            ("simulation", SimulationConfig, False),
            ("trainer", TrainerConfig, False),
            ("wandb", WandbConfig, True),
        )
        def train(cfg: DictConfig):
            # All specified fields are validated
    """

    def decorator(func: Callable) -> Callable:
        # Apply validators in reverse order so they execute in the order specified
        decorated = func
        for field_name, config_class, optional in reversed(validations):
            decorated = validate_subconfig(field_name, config_class, optional)(decorated)
        return decorated

    return decorator


def auto_validate_config(func: Callable) -> Callable:
    """
    Decorator that automatically validates configs based on registered types.

    Requires config groups to be registered with config_registry.

    Usage:
        # Register config types
        config_registry.register("sim", SimulationConfig)
        config_registry.register("trainer", TrainerConfig)

        @hydra.main(version_base=None, config_path="../configs", config_name="train")
        @auto_validate_config
        def train(cfg: DictConfig):
            # All registered config groups in cfg are automatically validated
    """

    @functools.wraps(func)
    def wrapper(cfg: Any) -> Any:
        if isinstance(cfg, ListConfig):
            raise TypeError(
                "auto_validate_config decorator received ListConfig instead of DictConfig. "
                "This usually means the config file contains a list at the root level."
            )

        errors = {}

        # Check each field in config
        for field_name in cfg:
            config_class = config_registry.get(field_name)
            if config_class:
                try:
                    field_value = cfg[field_name]
                    if isinstance(field_value, ListConfig):
                        errors[field_name] = f"Field is a ListConfig, cannot validate with {config_class.__name__}"
                        continue

                    validated = config_class(field_value)
                    cfg[field_name] = validated.model_dump()
                except ValidationError as e:
                    errors[field_name] = str(e)

        if errors:
            error_msg = "Configuration validation failed:\n"
            for field, error in errors.items():
                error_msg += f"\n{field}:\n{error}\n"
            raise ValidationError(error_msg)

        return func(cfg)

    return wrapper


class ValidatedConfig:
    """
    Context manager for validated configs.

    Provides a way to validate configs in existing code without decorators.

    Usage:
        with ValidatedConfig(cfg.simulation, SimulationConfig) as sim_cfg:
            # Use sim_cfg as a validated SimulationConfig instance
            run_simulation(sim_cfg)
    """

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]], config_class: Type[T], strict: bool = True):
        if isinstance(cfg, ListConfig):
            raise TypeError(
                f"ValidatedConfig received ListConfig instead of DictConfig. "
                f"Cannot validate a list with {config_class.__name__}."
            )

        self.cfg = cfg
        self.config_class = config_class
        self.strict = strict
        self.validated_cfg: Optional[T] = None

    def __enter__(self) -> T:
        try:
            self.validated_cfg = self.config_class(self.cfg)
            return self.validated_cfg
        except ValidationError as e:
            if self.strict:
                raise
            print(f"Warning: Configuration validation failed: {e}")
            # Return a best-effort instance using model_construct (skips validation)
            return self.config_class.model_construct(**self.cfg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def extract_validated_configs(cfg: DictConfig, **field_mappings: Type[BaseConfig]) -> Dict[str, BaseConfig]:
    """
    Extract and validate multiple config fields at once.

    Args:
        cfg: The main config
        **field_mappings: Mapping of field names to config classes

    Returns:
        Dictionary of validated config instances

    Usage:
        configs = extract_validated_configs(
            cfg,
            simulation=SimulationConfig,
            trainer=TrainerConfig,
            wandb=WandbConfig,
        )

        sim_cfg = configs["simulation"]  # Validated SimulationConfig
    """
    if isinstance(cfg, ListConfig):
        raise TypeError(
            "extract_validated_configs received ListConfig instead of DictConfig. "
            "This function requires a dictionary-like config."
        )

    validated = {}
    errors = {}

    for field_name, config_class in field_mappings.items():
        try:
            # Handle nested fields
            field_parts = field_name.split(".")
            current = cfg
            for i, part in enumerate(field_parts):
                current = current[part]
                # Check if we hit a ListConfig along the way
                if isinstance(current, ListConfig) and i < len(field_parts) - 1:
                    raise TypeError(
                        f"Encountered ListConfig at '{'.'.join(field_parts[: i + 1])}' "
                        f"while navigating to '{field_name}'. Cannot access dict keys in a list."
                    )

            # Final value shouldn't be a ListConfig either
            if isinstance(current, ListConfig):
                raise TypeError(f"Field '{field_name}' is a ListConfig, cannot validate with {config_class.__name__}")

            validated[field_name] = config_class(current)
        except (KeyError, AttributeError) as e:
            errors[field_name] = f"Field not found in config: {e}"
        except TypeError as e:
            errors[field_name] = str(e)
        except ValidationError as e:
            errors[field_name] = str(e)

    if errors:
        error_msg = "Failed to extract and validate configs:\n"
        for field, error in errors.items():
            error_msg += f"\n{field}: {error}\n"
        raise ValidationError(error_msg)

    return validated
