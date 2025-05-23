from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_moduly import MettaModule


class ComponentContainer(nn.ModuleDict):
    """Enhanced version of MettaAgent.components preserving recursive execution.

    This container replaces the existing MettaAgent.components ModuleDict with enhanced
    functionality for dependency tracking and recursive execution, while preserving
    the hotswapping interface and call-by-name patterns.
    """

    def __init__(self):
        super().__init__()
        self.dependencies = {}  # Track component dependencies
        self._execution_cache = {}  # Cache computed results within forward pass

    def register_component(self, name: str, module: MettaModule, dependencies: Optional[List[str]] = None):
        """Register component with explicit dependencies.

        Args:
            name (str): Component name for registry and dependency resolution
            module (MettaModule): The module to register
            dependencies (List[str] | None): List of component names this module depends on
        """
        self[name] = module  # Standard nn.ModuleDict storage
        self.dependencies[name] = dependencies or []

    def forward(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Recursive execution preserving current elegant pattern.

        Args:
            component_name (str): Name of component to execute
            tensordict (TensorDict): Data container

        Returns:
            TensorDict: Updated tensordict with component outputs
        """
        # Check if already computed in this forward pass (caching)
        if component_name in self._execution_cache:
            return tensordict

        component = self[component_name]

        # Check if all outputs already exist (avoid recomputation)
        if hasattr(component, "out_keys") and component.out_keys:
            # Simple membership check without trying to iterate over keys view
            outputs_exist = True
            for out_key in component.out_keys:
                if out_key not in tensordict:
                    outputs_exist = False
                    break
            if outputs_exist:
                self._execution_cache[component_name] = True
                return tensordict

        # Recursively compute dependencies first
        if component_name in self.dependencies:
            for dep_name in self.dependencies[component_name]:
                tensordict = self.forward(dep_name, tensordict)

        # Execute this component
        tensordict = component(tensordict)
        self._execution_cache[component_name] = True

        return tensordict

    def replace_component(self, name: str, new_module: MettaModule, dependencies: Optional[List[str]] = None):
        """Preserve hotswapping capability.

        Args:
            name (str): Component name to replace
            new_module (MettaModule): New module implementation
            dependencies (List[str] | None): Updated dependencies if different
        """
        self[name] = new_module
        if dependencies is not None:
            self.dependencies[name] = dependencies

    def clear_cache(self):
        """Clear execution cache (called at start of each forward pass)."""
        self._execution_cache.clear()

    def get_component_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a specific component.

        Args:
            name (str): Component name

        Returns:
            List[str]: List of dependency names
        """
        return self.dependencies.get(name, [])

    def get_execution_order(self, target_component: str) -> List[str]:
        """Get execution order for reaching target component.

        Args:
            target_component (str): Component to reach

        Returns:
            List[str]: Ordered list of components to execute
        """
        visited = set()
        order = []

        def visit(component_name: str):
            if component_name in visited:
                return
            visited.add(component_name)

            # Visit dependencies first
            for dep in self.dependencies.get(component_name, []):
                visit(dep)

            order.append(component_name)

        visit(target_component)
        return order

    def validate_dependencies(self):
        """Validate that all dependencies exist and check for cycles.

        Raises:
            ValueError: If dependency validation fails
        """
        # Check all dependencies exist
        for component_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self:
                    raise ValueError(f"Component '{component_name}' depends on non-existent component '{dep}'")

        # Check for cycles using DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {name: WHITE for name in self.keys()}

        def has_cycle(node: str) -> bool:
            if colors[node] == GRAY:
                return True  # Found back edge (cycle)
            if colors[node] == BLACK:
                return False  # Already processed

            colors[node] = GRAY
            for dep in self.dependencies.get(node, []):
                if has_cycle(dep):
                    return True
            colors[node] = BLACK
            return False

        for component_name in list(self.keys()):
            if colors[component_name] == WHITE:
                if has_cycle(component_name):
                    raise ValueError(f"Circular dependency detected involving component '{component_name}'")

    def __repr__(self) -> str:
        """Human-readable representation."""
        lines = [f"ComponentContainer with {len(self)} components:"]
        for name, component in self.items():
            deps = self.dependencies.get(name, [])
            lines.append(f"  {name}: {type(component).__name__} (deps: {deps})")
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo of ComponentContainer
    from metta.agent.lib.metta_moduly import LinearModule

    print("=== ComponentContainer Demo (Agent-level registry) ===")

    # Agent-level component registry
    components = ComponentContainer()

    # Register components with dependencies
    components.register_component("obs_processor", LinearModule(10, 8, "observation", "processed_obs"))
    components.register_component(
        "policy", LinearModule(8, 3, "processed_obs", "action"), dependencies=["obs_processor"]
    )

    print(components)

    # Test recursive execution
    td = TensorDict({"observation": torch.randn(2, 10)}, batch_size=2)
    components.clear_cache()  # Start fresh execution
    result = components.forward("policy", td)  # Will recursively execute obs_processor first

    result_keys = [str(k) for k in result.keys()]  # Convert to strings to avoid linter issues
    print(f"Result contains: {result_keys}")
    print(f"Action shape: {result['action'].shape}")

    # Test execution order
    print(f"Execution order for 'policy': {components.get_execution_order('policy')}")

    # Test dependency validation
    components.validate_dependencies()
    print("Dependencies validated successfully")
