"""Simulation registry - programmatic replacement for sim/*.yaml files."""

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig


@dataclass
class SimulationSpec:
    """Specification for a simulation."""

    name: str
    env: str
    num_episodes: int = 1
    max_time_s: int = 120
    env_overrides: dict = None
    npc_policy_uri: Optional[str] = None
    policy_agents_pct: float = 1.0

    def to_config(self) -> SingleEnvSimulationConfig:
        """Convert to simulation config."""
        return SingleEnvSimulationConfig(
            env=self.env,
            num_episodes=self.num_episodes,
            max_time_s=self.max_time_s,
            env_overrides=self.env_overrides or {},
            npc_policy_uri=self.npc_policy_uri,
            policy_agents_pct=self.policy_agents_pct,
        )


class SimulationRegistry:
    """Registry for simulation configurations.

    This replaces the large YAML files with a programmatic interface
    that allows for dynamic simulation registration and composition.
    """

    def __init__(self):
        self._simulations: Dict[str, SimulationSpec] = {}
        self._suites: Dict[str, List[str]] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, spec: SimulationSpec, categories: Optional[List[str]] = None) -> None:
        """Register a simulation specification."""
        self._simulations[spec.name] = spec

        # Extract category from name if not provided
        if categories is None and "/" in spec.name:
            category = spec.name.split("/")[0]
            categories = [category]

        # Add to categories
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(spec.name)

    def register_suite(self, name: str, simulation_names: List[str]) -> None:
        """Register a named suite of simulations."""
        self._suites[name] = simulation_names

    def get(self, name: str) -> SimulationSpec:
        """Get a simulation by name."""
        if name not in self._simulations:
            raise KeyError(f"Unknown simulation: {name}")
        return copy.deepcopy(self._simulations[name])

    def get_suite(self, name: str) -> SimulationSuiteConfig:
        """Get a simulation suite configuration."""
        if name == "all":
            # Special case - return all simulations
            simulation_names = list(self._simulations.keys())
        elif name in self._suites:
            simulation_names = self._suites[name]
        elif name in self._categories:
            # Allow category names as suites
            simulation_names = self._categories[name]
        else:
            raise KeyError(f"Unknown suite: {name}")

        simulations = {}
        for sim_name in simulation_names:
            spec = self.get(sim_name)
            simulations[sim_name] = spec.to_config()

        return SimulationSuiteConfig(
            name=name,
            num_episodes=1,
            max_time_s=120,
            simulations=simulations,
        )

    def list_simulations(self) -> List[str]:
        """List all registered simulations."""
        return sorted(self._simulations.keys())

    def list_suites(self) -> List[str]:
        """List all registered suites."""
        return sorted(self._suites.keys()) + ["all"]

    def list_categories(self) -> List[str]:
        """List all categories."""
        return sorted(self._categories.keys())

    def filter(self, predicate: Callable[[SimulationSpec], bool]) -> List[str]:
        """Filter simulations by a predicate."""
        return [name for name, spec in self._simulations.items() if predicate(spec)]


# Global registry instance
_registry = SimulationRegistry()


def register_simulation(name: str, env: str, num_episodes: int = 1, max_time_s: int = 120, **kwargs) -> None:
    """Register a simulation in the global registry."""
    spec = SimulationSpec(name=name, env=env, num_episodes=num_episodes, max_time_s=max_time_s, **kwargs)
    _registry.register(spec)


def get_simulation_suite(name: str) -> SimulationSuiteConfig:
    """Get a simulation suite from the global registry."""
    return _registry.get_suite(name)


def get_registry() -> SimulationRegistry:
    """Get the global simulation registry."""
    return _registry


# Register default simulations (equivalent to sim/all.yaml)
def register_default_simulations():
    """Register the default set of simulations."""

    # Navigation simulations
    navigation_sims = [
        ("navigation/emptyspace_withinsight", "env/mettagrid/navigation/evals/emptyspace_withinsight"),
        ("navigation/emptyspace_outofsight", "env/mettagrid/navigation/evals/emptyspace_outofsight"),
        ("navigation/emptyspace_sparse", "env/mettagrid/navigation/evals/emptyspace_sparse"),
        ("navigation/walls_withinsight", "env/mettagrid/navigation/evals/walls_withinsight"),
        ("navigation/walls_outofsight", "env/mettagrid/navigation/evals/walls_outofsight"),
        ("navigation/walls_sparse", "env/mettagrid/navigation/evals/walls_sparse"),
        ("navigation/cylinder", "env/mettagrid/navigation/evals/cylinder"),
        ("navigation/obstacles0", "env/mettagrid/navigation/evals/obstacles0"),
        ("navigation/obstacles1", "env/mettagrid/navigation/evals/obstacles1"),
        ("navigation/obstacles2", "env/mettagrid/navigation/evals/obstacles2"),
        ("navigation/obstacles3", "env/mettagrid/navigation/evals/obstacles3"),
        ("navigation/corridors", "env/mettagrid/navigation/evals/corridors"),
        ("navigation/labyrinth", "env/mettagrid/navigation/evals/labyrinth"),
        ("navigation/radialmaze", "env/mettagrid/navigation/evals/radialmaze"),
    ]

    for name, env in navigation_sims:
        register_simulation(name, env)

    # Object use simulations
    objectuse_sims = [
        ("objectuse/altar_use_free", "env/mettagrid/object_use/evals/altar_use_free"),
        ("objectuse/armory_use_free", "env/mettagrid/object_use/evals/armory_use_free"),
        ("objectuse/armory_use", "env/mettagrid/object_use/evals/armory_use"),
        ("objectuse/generator_use_free", "env/mettagrid/object_use/evals/generator_use_free"),
        ("objectuse/generator_use", "env/mettagrid/object_use/evals/generator_use"),
        ("objectuse/lasery_use_free", "env/mettagrid/object_use/evals/lasery_use_free"),
        ("objectuse/lasery_use", "env/mettagrid/object_use/evals/lasery_use"),
        ("objectuse/mine_use", "env/mettagrid/object_use/evals/mine_use"),
        ("objectuse/shoot_out", "env/mettagrid/object_use/evals/shoot_out"),
        ("objectuse/swap_in", "env/mettagrid/object_use/evals/swap_in"),
        ("objectuse/swap_out", "env/mettagrid/object_use/evals/swap_out"),
    ]

    for name, env in objectuse_sims:
        register_simulation(name, env, policy_agents_pct=1.0)

    # Memory simulations
    memory_sims = [
        ("memory/easy", "env/mettagrid/memory/evals/easy"),
        ("memory/medium", "env/mettagrid/memory/evals/medium"),
        ("memory/hard", "env/mettagrid/memory/evals/hard"),
        ("memory/access_cross", "env/mettagrid/memory/evals/access_cross"),
        ("memory/boxout", "env/mettagrid/memory/evals/boxout"),
        ("memory/choose_wisely", "env/mettagrid/memory/evals/choose_wisely"),
        ("memory/corners", "env/mettagrid/memory/evals/corners"),
    ]

    for name, env in memory_sims:
        register_simulation(name, env)

    # Register common suites
    _registry.register_suite("navigation", [name for name, _ in navigation_sims])
    _registry.register_suite("objectuse", [name for name, _ in objectuse_sims])
    _registry.register_suite("memory", [name for name, _ in memory_sims])

    # Quick test suite
    _registry.register_suite(
        "quick",
        [
            "navigation/emptyspace_withinsight",
            "objectuse/altar_use_free",
            "memory/easy",
        ],
    )
