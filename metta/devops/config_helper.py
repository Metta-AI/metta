#!/usr/bin/env python3
"""
Helper script for skypilot and sandbox to get configuration.

This can be used to:
1. Export environment variables for scripts
2. Get specific configuration values
3. Initialize environments with proper config
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import metta modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metta.config.schema import get_config


def export_env_vars():
    """Export all configuration as shell environment variables."""
    config = get_config()
    env_vars = config.export_env_vars()

    for key, value in env_vars.items():
        print(f"export {key}='{value}'")


def export_env_file(output_file: str):
    """Export configuration to .env file format."""
    config = get_config()
    env_vars = config.export_env_vars()

    with open(output_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print(f"Configuration exported to {output_file}")


def get_value(key: str):
    """Get a specific configuration value."""
    config = get_config()

    # Navigate nested keys (e.g., "wandb.entity")
    parts = key.split(".")
    value = config

    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        else:
            print(f"Error: Key '{key}' not found", file=sys.stderr)
            sys.exit(1)

    # Handle different types
    if value is None:
        print("")
    elif isinstance(value, bool):
        print("true" if value else "false")
    else:
        print(value)


def export_json():
    """Export configuration as JSON."""
    config = get_config()

    # Convert to dict
    data = {
        "wandb": config.wandb.__dict__,
        "observatory": {k: v for k, v in config.observatory.__dict__.items() if k != "auth_token"},
        "storage": config.storage.__dict__,
        "datadog": {k: v for k, v in config.datadog.__dict__.items() if not k.endswith("_key")},
        "profile": config.profile,
    }

    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Metta configuration helper for DevOps scripts")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export commands
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument(
        "--format", choices=["env", "json", "file"], default="env", help="Export format (default: env)"
    )
    export_parser.add_argument("--output", help="Output file (for format=file)")

    # Get specific value
    get_parser = subparsers.add_parser("get", help="Get specific configuration value")
    get_parser.add_argument("key", help="Configuration key (e.g., wandb.entity)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "export":
        if args.format == "env":
            export_env_vars()
        elif args.format == "json":
            export_json()
        elif args.format == "file":
            if not args.output:
                print("Error: --output required for format=file", file=sys.stderr)
                sys.exit(1)
            export_env_file(args.output)
    elif args.command == "get":
        get_value(args.key)


if __name__ == "__main__":
    main()
