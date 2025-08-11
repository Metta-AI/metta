"""MettaGrid configuration module.

This module provides all configuration classes and utilities for MettaGrid environments:
- Core Pydantic configuration models
- Predefined common objects
- Builder functions for creating configurations easily
"""

# Core configuration classes
# Predefined objects for common use
from metta.mettagrid.config import objects

# Builder functions for easy configuration creation
from metta.mettagrid.config.builder import arena, combat_arena, empty_arena, simple_arena
from metta.mettagrid.config.mettagrid_config import (
    EnvConfig,
    PyActionConfig,
    PyActionsConfig,
    PyAgentConfig,
    PyAgentRewards,
    PyAttackActionConfig,
    PyChangeGlyphActionConfig,
    PyConverterConfig,
    PyGameConfig,
    PyGlobalObsConfig,
    PyGroupConfig,
    PyInventoryRewards,
    PyStatsRewards,
    PyWallConfig,
)

__all__ = [
    # Core config classes
    "EnvConfig",
    "PyGameConfig",
    "PyAgentConfig",
    "PyAgentRewards",
    "PyInventoryRewards",
    "PyStatsRewards",
    "PyGroupConfig",
    "PyActionsConfig",
    "PyActionConfig",
    "PyAttackActionConfig",
    "PyChangeGlyphActionConfig",
    "PyGlobalObsConfig",
    "PyWallConfig",
    "PyConverterConfig",
    # Objects module
    "objects",
    # Builder functions
    "arena",
    "simple_arena",
    "combat_arena",
    "empty_arena",
]
