from collections import defaultdict, deque
from typing import Dict, List, Optional

import torch.nn as nn
from tensordict import TensorDict

from .metta_module import MettaModule


class ModularNetwork(nn.Module):
    """Component container responsible for orchestrating data flow and tracking dependencies.

    Key features:
    - Key-based dataflow: Components declare semantic dependencies via data keys
    - Automatic dependency resolution: Execution order determined by data dependencies
    - Hot-swappable components: Any component producing the same keys can be swapped
    - PyTorch compatibility: Full support for all PyTorch operations

    Example:
        # Create network with automatic dependency resolution
        network = ModularNetwork()

        # Add components - order doesn't matter, dependencies auto-resolved
        network.add_component("policy", LinearModule(128, 4, "features", "action_logits"))
        network.add_component("encoder", LinearModule(64, 128, "observation", "features"))

        # Execute with automatic ordering: encoder → policy
        td = TensorDict({"observation": torch.randn(10, 64)}, batch_size=[10])
        result = network(td)  # Contains "features" and "action_logits"
    """

    def __init__(self, components: Optional[Dict[str, MettaModule]] = None):
        """Initialize modular network.

        Args:
            components: Optional dict of {name: component} to initialize with
        """
        super().__init__()

        # Use nn.ModuleDict for proper PyTorch integration
        self.components = nn.ModuleDict()

        # Track data flow for dependency resolution
        self._key_to_producer = {}  # {output_key: component_name}
        self._component_dependencies = defaultdict(set)  # {component_name: {required_keys}}
        self._execution_order = []  # Cached topological sort
        self._order_dirty = True  # Flag to recompute execution order

        # Initialize with provided components
        if components:
            for name, component in components.items():
                self.add_component(name, component)

    def add_component(self, name: str, component: MettaModule):
        """Add a component to the network with automatic dependency tracking.

        Args:
            name: Unique identifier for the component in this network
            component: MettaModule to add

        Raises:
            ValueError: If component name already exists or creates circular dependencies
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already exists")

        # Check for output key conflicts
        for out_key in component.out_keys:
            if out_key in self._key_to_producer:
                existing_producer = self._key_to_producer[out_key]
                raise ValueError(
                    f"Output key '{out_key}' conflict: component '{name}' and "
                    f"'{existing_producer}' both produce this key"
                )

        # Register component
        self.components[name] = component

        # Update dependency tracking
        for out_key in component.out_keys:
            self._key_to_producer[out_key] = name

        self._component_dependencies[name] = set(component.in_keys)
        self._order_dirty = True

        # Validate no circular dependencies
        try:
            self._compute_execution_order()
        except ValueError as e:
            # Rollback on circular dependency
            self.remove_component(name)
            raise e

    def remove_component(self, name: str):
        """Remove a component and update dependency tracking.

        Args:
            name: Name of component to remove

        Raises:
            KeyError: If component doesn't exist
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not found")

        component = self.components[name]

        # Check if other components depend on this one's outputs
        dependent_components = []
        for out_key in component.out_keys:
            for comp_name, deps in self._component_dependencies.items():
                if comp_name != name and out_key in deps:
                    dependent_components.append(comp_name)

        if dependent_components:
            raise ValueError(
                f"Cannot remove '{name}': components {dependent_components} depend on its outputs {component.out_keys}"
            )

        # Remove from tracking
        del self.components[name]
        for out_key in component.out_keys:
            del self._key_to_producer[out_key]
        del self._component_dependencies[name]
        self._order_dirty = True

    def swap_component(self, name: str, new_component: MettaModule):
        """Hot-swap a component while preserving dependencies.

        The new component must have compatible in_keys and out_keys.

        Args:
            name: Name of existing component to replace
            new_component: New component with compatible interface

        Raises:
            ValueError: If key compatibility is broken
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not found")

        old_component = self.components[name]

        # Validate compatibility
        if set(new_component.out_keys) != set(old_component.out_keys):
            raise ValueError(f"Output key mismatch: old {old_component.out_keys} vs new {new_component.out_keys}")

        # Update component (preserves dependencies automatically)
        self.components[name] = new_component
        self._component_dependencies[name] = set(new_component.in_keys)
        self._order_dirty = True

    def _compute_execution_order(self) -> List[str]:
        """Compute topological sort for execution order based on dependencies.

        Returns:
            List of component names in execution order

        Raises:
            ValueError: If circular dependencies exist
        """
        if not self._order_dirty and self._execution_order:
            return self._execution_order

        # Build dependency graph
        graph = defaultdict(set)  # {component: set_of_components_it_depends_on}
        in_degree = defaultdict(int)

        # Initialize all components
        for comp_name in self.components.keys():
            in_degree[comp_name] = 0

        # Build edges based on data dependencies
        for comp_name, required_keys in self._component_dependencies.items():
            for req_key in required_keys:
                if req_key in self._key_to_producer:
                    producer = self._key_to_producer[req_key]
                    if producer != comp_name:  # No self-dependencies
                        graph[producer].add(comp_name)
                        in_degree[comp_name] += 1

        # Kahn's algorithm for topological sort
        queue = deque([comp for comp, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Reduce in-degree for dependents
            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for circular dependencies
        if len(result) != len(self.components):
            remaining = set(self.components.keys()) - set(result)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        self._execution_order = result
        self._order_dirty = False
        return result

    def forward(self, td: TensorDict) -> TensorDict:
        """Execute all components in dependency order.

        Args:
            td: Input TensorDict containing required input keys

        Returns:
            TensorDict with all component outputs added

        Raises:
            KeyError: If required input keys are missing
        """
        execution_order = self._compute_execution_order()

        # Validate required inputs are present
        missing_keys = []
        for comp_name in execution_order:
            component = self.components[comp_name]
            for in_key in component.in_keys:
                if in_key not in td and in_key not in self._key_to_producer:
                    missing_keys.append((comp_name, in_key))

        if missing_keys:
            raise KeyError(f"Missing required inputs: {missing_keys}")

        # Execute components in order
        for comp_name in execution_order:
            component = self.components[comp_name]
            td = component(td)

        return td

    def get_component(self, name: str) -> MettaModule:
        """Get component by name.

        Args:
            name: Component name

        Returns:
            The component

        Raises:
            KeyError: If component not found
        """
        return self.components[name]  # type: ignore

    def get_execution_order(self) -> List[str]:
        """Get current execution order.

        Returns:
            List of component names in execution order
        """
        return self._compute_execution_order().copy()

    def get_dependencies(self, component_name: str) -> Dict[str, str]:
        """Get dependencies for a component.

        Args:
            component_name: Name of component

        Returns:
            Dict mapping {required_key: producer_component_name}
        """
        if component_name not in self.components:
            raise KeyError(f"Component '{component_name}' not found")

        component = self.components[component_name]
        dependencies = {}

        for in_key in component.in_keys:
            if in_key in self._key_to_producer:
                dependencies[in_key] = self._key_to_producer[in_key]
            else:
                dependencies[in_key] = "<external>"  # Must be provided in input

        return dependencies

    def get_data_flow_graph(self) -> Dict[str, Dict]:
        """Get complete data flow visualization.

        Returns:
            Dict with dependency information for debugging/visualization
        """
        graph = {}

        for comp_name in self.components:
            component = self.components[comp_name]
            graph[comp_name] = {
                "component": component.__class__.__name__,
                "in_keys": component.in_keys,
                "out_keys": component.out_keys,
                "dependencies": self.get_dependencies(comp_name),
                "dependents": [],
            }

        # Fill in dependents
        for comp_name, info in graph.items():
            for dep_key, producer in info["dependencies"].items():
                if producer != "<external>" and producer in graph:
                    graph[producer]["dependents"].append(comp_name)

        return graph

    def __len__(self) -> int:
        """Number of components in the network."""
        return len(self.components)

    def __contains__(self, name: str) -> bool:
        """Check if component exists."""
        return name in self.components

    def __iter__(self):
        """Iterate over component names."""
        return iter(self.components)

    def __repr__(self) -> str:
        """String representation showing components and dependencies."""
        lines = [f"ModularNetwork({len(self.components)} components):"]

        try:
            execution_order = self.get_execution_order()
            for i, comp_name in enumerate(execution_order):
                component = self.components[comp_name]
                deps = self.get_dependencies(comp_name)
                dep_str = ", ".join([f"{k}←{v}" for k, v in deps.items() if v != "<external>"])
                lines.append(f"  {i + 1}. {comp_name}: {component.__class__.__name__}")
                if dep_str:
                    lines.append(f"      deps: {dep_str}")
                lines.append(f"      keys: {component.in_keys} → {component.out_keys}")
        except ValueError as e:
            lines.append(f"  ERROR: {e}")

        return "\n".join(lines)
