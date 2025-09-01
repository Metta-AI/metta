"""
Unified configuration system for Metta.

This module provides:
- A single configuration file (~/.metta/config.yaml) for all settings
- Component-based configuration interfaces
- Environment variable export for DevOps integration
- Profile-based defaults
"""

from metta.config.schema import MettaConfig, get_config, reload_config

__all__ = ["MettaConfig", "get_config", "reload_config"]
