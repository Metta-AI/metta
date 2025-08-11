"""Utility functions for running tools with configuration."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def run_tool(tool_name: str, config: Any, path: str | Path) -> subprocess.CompletedProcess:
    """Save config as YAML and run tool with --config pointing to that YAML.

    Args:
        tool_name: Name of the tool to run (e.g., "train")
        config: Pydantic configuration object to save as YAML
        path: Directory path where tool_config.yaml will be saved

    Returns:
        CompletedProcess from running the tool

    Raises:
        TypeError: If config is not a Pydantic model
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    config_path = path / "tool_config.yaml"

    # Convert Pydantic model to dict
    config_dict = config.model_dump()

    # Save config as indented YAML
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)

    # Run the tool with the config
    tool_path = Path(__file__).parent.parent.parent / "tools" / f"{tool_name}.py"
    cmd = [sys.executable, str(tool_path), f"--config={config_path}"]

    return subprocess.run(cmd, check=True)
