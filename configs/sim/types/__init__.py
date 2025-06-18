"""
Configuration types for simulation module.

This module provides strongly-typed configuration classes that:
1. Validate Hydra/OmegaConf configurations at runtime
2. Provide IDE support and type checking
3. Document expected configuration structure
"""

from metta.sim.types.simulation_config import (
    SimulationConfig,
    SimulationSuiteConfig,
    SingleEnvSimulationConfig,
)

__all__ = [
    "SimulationConfig",
    "SingleEnvSimulationConfig",
    "SimulationSuiteConfig",
]
