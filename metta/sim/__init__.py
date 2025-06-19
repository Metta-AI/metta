"""Simulation and evaluation interfaces for Metta."""

# Registry interfaces
from .registry import (
    SimulationRegistry,
    SimulationSpec,
    get_registry,
    get_simulation_suite,
    register_default_simulations,
    register_simulation,
)
from .simulation import Simulation
from .simulation_config import (
    SimulationConfig,
    SimulationSuiteConfig,
    SingleEnvSimulationConfig,
)
from .simulation_suite import SimulationSuite

__all__ = [
    # Core simulation
    "Simulation",
    "SimulationConfig",
    "SingleEnvSimulationConfig",
    "SimulationSuiteConfig",
    "SimulationSuite",
    # Registry
    "SimulationRegistry",
    "SimulationSpec",
    "register_simulation",
    "get_simulation_suite",
    "get_registry",
    "register_default_simulations",
]

