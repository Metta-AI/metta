"""Utilities for parsing config files in argparse."""

import argparse
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def config_file_parser(config_class: Type[T]) -> Type[argparse.Action]:
    """Create an argparse Action that parses a YAML file into a Pydantic model.

    Args:
        config_class: The Pydantic model class to parse the YAML into

    Returns:
        An argparse Action class that handles the parsing
    """

    class ConfigFileAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            config_path = Path(values)
            if not config_path.exists():
                raise argparse.ArgumentTypeError(f"Config file not found: {values}")

            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)

                # Parse the config data into the Pydantic model
                config = config_class.model_validate(config_data)
                setattr(namespace, self.dest, config)
            except yaml.YAMLError as e:
                raise argparse.ArgumentTypeError(f"Invalid YAML in config file: {e}") from e
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Error parsing config file: {e}") from e

    return ConfigFileAction


def optional_config_file_parser(config_class: Type[T]) -> Type[argparse.Action]:
    """Create an argparse Action that optionally parses a YAML file into a Pydantic model.

    If no file is provided, returns None.

    Args:
        config_class: The Pydantic model class to parse the YAML into

    Returns:
        An argparse Action class that handles the parsing
    """

    class OptionalConfigFileAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values is None:
                setattr(namespace, self.dest, None)
                return

            config_path = Path(values)
            if not config_path.exists():
                raise argparse.ArgumentTypeError(f"Config file not found: {values}")

            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)

                # Parse the config data into the Pydantic model
                config = config_class.model_validate(config_data)
                setattr(namespace, self.dest, config)
            except yaml.YAMLError as e:
                raise argparse.ArgumentTypeError(f"Invalid YAML in config file: {e}") from e
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Error parsing config file: {e}") from e

    return OptionalConfigFileAction
