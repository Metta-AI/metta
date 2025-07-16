#!/usr/bin/env python3

# TODO: currently this script just prints the config, but it should also validate it

"""
Config validation script that loads and prints Hydra configurations.
"""

import argparse
import sys

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.config import config_from_path


def load_and_print_config(
    config_path: str,
    overrides: list[str] | None = None,
    exit_on_failure: bool = True,
    print_cfg: bool = True,
) -> DictConfig | ListConfig | None:
    """
    Load a Hydra configuration and print it as YAML.

    Args:
        config_path: Path to the configuration file (relative to configs/ directory)
    """
    # Strip "configs/" prefix if present
    if config_path.startswith("configs/"):
        config_path = config_path[len("configs/") :]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Try the existing utility first for complex configs
    if any(x in config_path for x in ["env/mettagrid", "trainer"]):
        with hydra.initialize(config_path="../configs", version_base=None):
            try:
                # Use the existing utility that handles path extraction and complex compositions
                cfg = config_from_path(config_path)
                if print_cfg:
                    # Print without resolving to avoid interpolation issues
                    yaml_output = OmegaConf.to_yaml(cfg, resolve=False)
                    print("# Note: Loaded using config_from_path utility, interpolations not resolved")
                    print(yaml_output)
                return cfg

            except Exception as e:
                # Fall back to the regular hydra compose method
                print(f"# Note: config_from_path failed ({e}), trying fallback approach", file=sys.stderr)

    # Initialize Hydra with the configs directory
    with hydra.initialize(config_path="../configs", version_base=None):
        try:
            # Provide basic overrides to handle missing required values
            if overrides is None:
                overrides = []

            # Add run override for configs that need it (like trainer)
            if "trainer" in config_path or any(x in config_path for x in ["sim_job", "train_job"]):
                overrides.append("+run=test_run")  # Use + prefix to add new key

            # Load the configuration with overrides
            cfg = hydra.compose(config_name=config_path, overrides=overrides)

            # Print the configuration as YAML, but don't resolve interpolations
            # for some configs that might have missing dependencies
            resolve = True
            try:
                yaml_output = OmegaConf.to_yaml(cfg, resolve=resolve)
            except Exception:
                # If resolving fails, try without resolving interpolations
                resolve = False
                yaml_output = OmegaConf.to_yaml(cfg, resolve=resolve)
                print(f"# Note: Some interpolations could not be resolved for '{config_path}'")
            if print_cfg:
                print(yaml_output)
            return cfg

        except Exception as e:
            print(f"Error loading config '{config_path}': {e}", file=sys.stderr)
            if exit_on_failure:
                sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Load and print Hydra configurations")
    parser.add_argument("config_path", help="Path to the configuration file (relative to configs/ directory)")

    args = parser.parse_args()
    load_and_print_config(args.config_path)


if __name__ == "__main__":
    main()
