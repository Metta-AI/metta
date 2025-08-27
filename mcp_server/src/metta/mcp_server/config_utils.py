"""Configuration management utilities for MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

from metta.common.util.fs import get_repo_root


def get_config_root() -> Path:
    """Get the root directory for configurations."""
    try:
        # First try: Use the robust repo root detection from common utils
        repo_root = get_repo_root()
        config_dir = repo_root / "configs"
        if config_dir.exists():
            return config_dir
        raise FileNotFoundError(f"Configs directory not found at {config_dir}")
    except SystemExit:
        # Fallback: If no git repo found, try relative to this file
        # This handles cases where MCP server is run outside a git repository
        this_file = Path(__file__)
        repo_root = this_file.parent.parent.parent.parent.parent  # Go up 5 levels to project root
        config_dir = repo_root / "configs"
        if config_dir.exists():
            return config_dir

        raise FileNotFoundError(
            "Could not find configs directory. "
            "Please run the MCP server from within the metta git repository, "
            "or ensure the repository structure is intact."
        ) from None


def get_available_config_types() -> List[str]:
    """Get list of available configuration types."""
    config_root = get_config_root()
    config_types = []

    # Add directory-based config types
    for item in config_root.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "tmp":
            config_types.append(item.name)

    # Add root-level config files (without extension)
    for item in config_root.iterdir():
        if item.is_file() and item.suffix == ".yaml":
            config_types.append(item.stem)

    return sorted(config_types)


def list_configs_for_type(config_type: str) -> List[Dict[str, Any]]:
    """List all configuration files for a given type."""
    config_root = get_config_root()
    configs = []

    # Check if it's a directory-based config type
    type_dir = config_root / config_type
    if type_dir.exists() and type_dir.is_dir():
        for config_file in type_dir.rglob("*.yaml"):
            # Skip files in certain directories
            if any(part.startswith(".") or part == "__pycache__" for part in config_file.parts):
                continue

            # Get relative path from the type directory
            rel_path = config_file.relative_to(type_dir)
            config_name = str(rel_path.with_suffix(""))

            configs.append(
                {
                    "name": config_name,
                    "path": str(config_file),
                    "relative_path": str(rel_path),
                    "description": _extract_description(config_file),
                    "size": config_file.stat().st_size if config_file.exists() else 0,
                }
            )

    # Check if it's a root-level config file
    root_config = config_root / f"{config_type}.yaml"
    if root_config.exists():
        configs.append(
            {
                "name": config_type,
                "path": str(root_config),
                "relative_path": f"{config_type}.yaml",
                "description": _extract_description(root_config),
                "size": root_config.stat().st_size,
            }
        )

    return sorted(configs, key=lambda x: x["name"])


def _extract_description(config_file: Path) -> str | None:
    """Extract description from config file comments or README."""
    try:
        # First try to find a README in the same directory
        readme_path = config_file.parent / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            # Extract first paragraph or line as description
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if lines:
                return lines[0].lstrip("# ")

        # Try to extract from YAML comments
        with open(config_file) as f:
            lines = f.readlines()

        # Look for the first comment line that looks like a description
        for line in lines:
            line = line.strip()
            if line.startswith("#") and len(line) > 3:
                desc = line[1:].strip()
                # Skip common non-description comments
                if not any(skip in desc.lower() for skip in ["@package", "defaults:", "hydra"]):
                    return desc

        return None
    except Exception:
        return None


def validate_config(config_path: str, overrides: List[str] = None) -> Dict[str, Any]:
    """Validate a Hydra configuration with optional overrides."""
    if overrides is None:
        overrides = []

    try:
        config_root = get_config_root()

        # Handle both absolute and relative paths
        if Path(config_path).is_absolute():
            full_path = Path(config_path)
        else:
            # Try relative to config root first
            full_path = config_root / config_path
            if not full_path.exists():
                # Try adding .yaml extension
                full_path = config_root / f"{config_path}.yaml"

        if not full_path.exists():
            return {
                "valid": False,
                "error": f"Configuration file not found: {config_path}",
                "config": None,
            }

        # Load base configuration
        base_config = OmegaConf.load(full_path)

        # Apply overrides
        if overrides:
            override_config = OmegaConf.create({})
            for override in overrides:
                try:
                    # Parse Hydra-style overrides (key=value, +key=value, ++key=value)
                    if override.startswith("++"):
                        key, value = override[2:].split("=", 1)
                        OmegaConf.set(override_config, key, _parse_value(value), force_add=True)
                    elif override.startswith("+"):
                        key, value = override[1:].split("=", 1)
                        OmegaConf.set(override_config, key, _parse_value(value))
                    else:
                        key, value = override.split("=", 1)
                        OmegaConf.set(override_config, key, _parse_value(value))
                except Exception as e:
                    return {
                        "valid": False,
                        "error": f"Invalid override '{override}': {str(e)}",
                        "config": None,
                    }

            # Merge configurations
            merged_config = OmegaConf.merge(base_config, override_config)
        else:
            merged_config = base_config

        # Convert to regular dict for JSON serialization
        config_dict = OmegaConf.to_container(merged_config, resolve=True)

        return {
            "valid": True,
            "error": None,
            "config": config_dict,
            "overrides_applied": overrides,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "config": None,
        }


def _parse_value(value_str: str) -> Any:
    """Parse a string value to appropriate Python type."""
    # Handle null/None
    if value_str.lower() in ["null", "none", "~"]:
        return None

    # Handle booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Try to parse as number
    try:
        if "." in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass

    # Handle lists and dicts (basic JSON-like parsing)
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            import ast

            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

    if value_str.startswith("{") and value_str.endswith("}"):
        try:
            import ast

            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

    # Return as string
    return value_str


def get_config_schema(config_type: str) -> Dict[str, Any]:
    """Get schema information for a configuration type."""
    # For now, we'll provide a basic schema by analyzing existing configs
    configs = list_configs_for_type(config_type)
    if not configs:
        return {
            "error": f"No configurations found for type: {config_type}",
            "schema": None,
        }

    # Analyze the first config to extract structure
    try:
        first_config = configs[0]
        config_data = OmegaConf.load(first_config["path"])

        # Convert to dict and analyze structure
        config_dict = OmegaConf.to_container(config_data, resolve=True)
        schema = _analyze_config_structure(config_dict)

        return {
            "config_type": config_type,
            "schema": schema,
            "sample_configs": [c["name"] for c in configs[:5]],  # First 5 as examples
            "total_configs": len(configs),
        }

    except Exception as e:
        return {
            "error": f"Failed to analyze config schema: {str(e)}",
            "schema": None,
        }


def _analyze_config_structure(config_dict: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """Recursively analyze configuration structure to build schema."""
    schema = {}

    for key, value in config_dict.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            schema[key] = {
                "type": "object",
                "path": current_path,
                "properties": _analyze_config_structure(value, current_path),
            }
        elif isinstance(value, list):
            item_type = type(value[0]).__name__ if value else "unknown"
            schema[key] = {
                "type": "array",
                "path": current_path,
                "item_type": item_type,
                "example_length": len(value),
            }
        else:
            schema[key] = {
                "type": type(value).__name__,
                "path": current_path,
                "example_value": value,
            }

    return schema
