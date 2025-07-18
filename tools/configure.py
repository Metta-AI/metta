#!/usr/bin/env -S uv run
"""Configuration management tool for metta.

This tool allows users to configure default values and expressions
that will be used by other metta tools.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import OmegaConf


class MettaConfigureTool:
    def __init__(self):
        self.config_dir = Path("configs/user")
        self.config_file = self.config_dir / ".metta_tool_config.yaml"
        self.config = self._load_config()

    def _load_config(self):
        """Load the configuration file or create an empty config."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config(self):
        """Save the configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Configuration saved to {self.config_file}")

    def _evaluate_expression(self, expression: str) -> str:
        """Evaluate a run name expression.
        
        Supports placeholders:
        - {user} - Current username
        - {date} - Current date in YYYY-MM-DD format
        - {time} - Current time in HHMMSS format
        - {timestamp} - Full timestamp YYYY-MM-DD-HHMMSS
        """
        now = datetime.now()
        replacements = {
            "{user}": os.environ.get("USER", "unknown"),
            "{date}": now.strftime("%Y-%m-%d"),
            "{time}": now.strftime("%H%M%S"),
            "{timestamp}": now.strftime("%Y-%m-%d-%H%M%S"),
        }
        
        result = expression
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result

    def set_default_run_name(self, expression: str):
        """Set the default run name expression."""
        # Validate the expression by trying to evaluate it
        try:
            example = self._evaluate_expression(expression)
            print(f"Default run name expression set to: {expression}")
            print(f"Example evaluation: {example}")
        except Exception as e:
            print(f"Error: Invalid expression - {e}")
            return 1

        if "train" not in self.config:
            self.config["train"] = {}
        
        self.config["train"]["default_run_name"] = expression
        self._save_config()
        return 0

    def get_default_run_name(self) -> str | None:
        """Get the default run name expression."""
        return self.config.get("train", {}).get("default_run_name")

    def show_config(self):
        """Display the current configuration."""
        if not self.config:
            print("No configuration set.")
            return

        print("Current configuration:")
        print(yaml.dump(self.config, default_flow_style=False))
        
        # Show evaluated examples for expressions
        if "train" in self.config and "default_run_name" in self.config["train"]:
            expression = self.config["train"]["default_run_name"]
            example = self._evaluate_expression(expression)
            print(f"Default run name expression evaluates to: {example}")

    def clear_config(self, key: str = None):
        """Clear configuration."""
        if key:
            # Clear specific key
            parts = key.split(".")
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    print(f"Key {key} not found in configuration.")
                    return 1
                current = current[part]
            
            if parts[-1] in current:
                del current[parts[-1]]
                print(f"Cleared {key}")
                self._save_config()
            else:
                print(f"Key {key} not found in configuration.")
                return 1
        else:
            # Clear all
            self.config = {}
            self._save_config()
            print("Cleared all configuration.")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Configure default values for metta tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set default run name with user and timestamp
  metta tool configure --set-run-name "{user}-test-{timestamp}"
  
  # Set default run name with just date
  metta tool configure --set-run-name "experiment-{date}"
  
  # Show current configuration
  metta tool configure --show
  
  # Clear a specific configuration
  metta tool configure --clear train.default_run_name
  
  # Clear all configuration
  metta tool configure --clear-all

Available placeholders for run name expressions:
  {user}      - Current username
  {date}      - Current date (YYYY-MM-DD)
  {time}      - Current time (HHMMSS)
  {timestamp} - Full timestamp (YYYY-MM-DD-HHMMSS)
""",
    )

    parser.add_argument(
        "--set-run-name",
        metavar="EXPRESSION",
        help="Set the default run name expression for train command",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )
    parser.add_argument(
        "--clear",
        metavar="KEY",
        help="Clear a specific configuration key (e.g., train.default_run_name)",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all configuration",
    )

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    tool = MettaConfigureTool()

    if args.set_run_name:
        return tool.set_default_run_name(args.set_run_name)
    elif args.show:
        tool.show_config()
        return 0
    elif args.clear:
        return tool.clear_config(args.clear)
    elif args.clear_all:
        return tool.clear_config()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())