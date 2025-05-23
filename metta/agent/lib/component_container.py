from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from metta.agent.lib.metta_moduly import MettaModule


class ComponentContainer(TensorDictModuleBase):
    """Base container with immediate registration and elegant recursive execution.

    This is the foundation layer that provides:
    - Simple component storage in nn.ModuleDict
    - Elegant recursive execution pattern
    - Basic dependency tracking
    - TensorDictModule compliance
    """

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict()  # Store actual component instances
        self.dependencies = {}  # Track component dependencies
        self.in_keys = []  # Determined from registered components
        self.out_keys = []  # Determined from registered components

    def register_component(self, name: str, component: MettaModule, dependencies: Optional[List[str]] = None):
        """Register an actual component instance immediately.

        Args:
            name: Component name for registry and dependency resolution
            component: Actual MettaModule instance (already initialized)
            dependencies: List of component names this module depends on
        """
        self.components[name] = component
        self.dependencies[name] = dependencies or []
        self._update_container_keys()

    def execute_component(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Recursive execution preserving current elegant pattern.

        This is the heart of the architecture - the beautiful recursive execution
        pattern that automatically handles dependency resolution and caching.
        """
        component = self.components[component_name]

        # Check if already computed (output presence caching)
        if all(out_key in tensordict for out_key in getattr(component, "out_keys", [])):
            return tensordict

        # Recursively compute dependencies first
        for dep_name in self.dependencies[component_name]:
            self.execute_component(dep_name, tensordict)

        # Execute this component
        return component(tensordict)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """TensorDictModule interface - execute all components."""
        for component_name in self._topological_sort():
            self.execute_component(component_name, tensordict)
        return tensordict

    def execute_network(self, tensordict: TensorDict) -> TensorDict:
        """Execute the full network - more explicit version of forward()."""
        return self.forward(tensordict)

    def replace_component(self, name: str, new_component: MettaModule):
        """Preserve hotswapping capability."""
        self.components[name] = new_component
        self._update_container_keys()

    def get_component_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a specific component."""
        return self.dependencies.get(name, [])

    def get_execution_order(self, target_component: str) -> List[str]:
        """Get execution order for reaching target component."""
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

    def _update_container_keys(self):
        """Update container's in_keys and out_keys based on registered components."""
        all_in_keys = set()
        all_out_keys = set()

        for component in self.components.values():
            # Ensure component has the required attributes
            if hasattr(component, "in_keys") and isinstance(component.in_keys, list):
                all_in_keys.update(component.in_keys)
            if hasattr(component, "out_keys") and isinstance(component.out_keys, list):
                all_out_keys.update(component.out_keys)

        # Container's inputs are keys not produced by any component
        self.in_keys = [key for key in all_in_keys if key not in all_out_keys]
        self.out_keys = list(all_out_keys)

    def _topological_sort(self) -> List[str]:
        """Get components in dependency order."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return

            temp_visited.add(component_name)
            for dep_name in self.dependencies.get(component_name, []):
                visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)
            result.append(component_name)

        for component_name in self.components.keys():
            if component_name not in visited:
                visit(component_name)

        return result

    def __repr__(self) -> str:
        """Human-readable representation."""
        lines = [f"ComponentContainer with {len(self.components)} components:"]
        for name, component in self.components.items():
            deps = self.dependencies.get(name, [])
            lines.append(f"  {name}: {type(component).__name__} (deps: {deps})")
        return "\n".join(lines)


class SafeComponentContainer(ComponentContainer):
    """Adds comprehensive validation to component registration and execution.

    This wrapper adds:
    - Component interface validation
    - Dependency existence checking
    - Circular dependency detection
    - Runtime shape validation
    """

    def register_component(self, name: str, component: MettaModule, dependencies: Optional[List[str]] = None):
        """Register component with comprehensive validation."""
        self._validate_component(name, component)
        self._validate_dependencies(name, dependencies or [])
        super().register_component(name, component, dependencies)

    def _validate_component(self, name: str, component: MettaModule):
        """Validate component has proper MettaModule interface."""
        if not isinstance(component, MettaModule):
            raise TypeError(f"Component '{name}' must be a MettaModule, got {type(component)}")

        if not hasattr(component, "in_keys") or not hasattr(component, "out_keys"):
            raise ValueError(f"Component '{name}' must have in_keys and out_keys attributes")

        if not isinstance(component.in_keys, list) or not isinstance(component.out_keys, list):
            raise ValueError(f"Component '{name}' in_keys and out_keys must be lists")

        # Check for output key conflicts
        for existing_name, existing_component in self.components.items():
            if hasattr(existing_component, "out_keys") and isinstance(existing_component.out_keys, list):
                overlap = set(component.out_keys) & set(existing_component.out_keys)
                if overlap:
                    raise ValueError(f"Output key conflict: {overlap} between '{name}' and '{existing_name}'")

    def _validate_dependencies(self, name: str, dependencies: List[str]):
        """Validate dependencies exist and check for cycles."""
        for dep_name in dependencies:
            if dep_name not in self.components and dep_name != name:
                raise ValueError(f"Dependency '{dep_name}' not found for component '{name}'")

        # Create temporary dependency structure and check for cycles
        temp_deps = self.dependencies.copy()
        temp_deps[name] = dependencies
        self._check_circular_dependencies(temp_deps)

    def _check_circular_dependencies(self, deps_dict: Dict[str, List[str]]):
        """Check for circular dependencies in dependency graph."""
        visited = set()
        temp_visited = set()

        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return

            temp_visited.add(component_name)
            for dep_name in deps_dict.get(component_name, []):
                if dep_name in deps_dict:  # Only visit if it's a registered component
                    visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)

        for component_name in deps_dict.keys():
            if component_name not in visited:
                visit(component_name)


class LazyComponentContainer(ComponentContainer):
    """Adds deferred initialization with automatic shape inference.

    This wrapper adds:
    - Configuration-based registration
    - Automatic shape inference during initialization
    - Deferred component instantiation
    - Custom shape inference patterns
    """

    def __init__(self):
        super().__init__()
        self.component_configs = {}  # Store component configurations
        self.initialized = False

    def register_component_config(
        self,
        name: str,
        component_class,
        config: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ):
        """Register component configuration for deferred initialization.

        Args:
            name: Component name for registry and dependency resolution
            component_class: Class to instantiate (e.g., LinearModule)
            config: Configuration dict (e.g., {"out_features": 128})
            dependencies: List of component names this module depends on
            in_keys: Input keys for this component
            out_keys: Output keys for this component
        """
        self.component_configs[name] = {
            "class": component_class,
            "config": config,
            "in_keys": in_keys or [],
            "out_keys": out_keys or [],
        }
        self.dependencies[name] = dependencies or []

        # Update container's keys based on registered component specs
        self._update_container_keys_from_configs()

    def initialize_with_input_shapes(self, input_shapes: Dict[str, tuple]):
        """Initialize all components with specified input shapes.

        Args:
            input_shapes: Dict mapping container's in_keys to their shapes (without batch dim)
                         e.g., {"observation": (64,)} for obs_dim=64
        """
        if self.initialized:
            return

        # Validate that all container inputs have shapes specified
        for in_key in self.in_keys:
            if in_key not in input_shapes:
                raise ValueError(f"Shape not specified for container input '{in_key}'")

        # Initialize components in dependency order
        shape_registry = input_shapes.copy()

        for component_name in self._topological_sort_configs():
            component = self._create_component_with_shapes(component_name, shape_registry)
            # Register the actual component using the base class method
            super().register_component(component_name, component, self.dependencies[component_name])

        self.initialized = True

    def execute_component(self, component_name: str, tensordict: TensorDict) -> TensorDict:
        """Recursive execution with initialization check."""
        if not self.initialized:
            raise RuntimeError("LazyComponentContainer must be initialized with input shapes first")
        return super().execute_component(component_name, tensordict)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """TensorDictModule interface with initialization check."""
        if not self.initialized:
            raise RuntimeError("LazyComponentContainer must be initialized with input shapes first")
        return super().forward(tensordict)

    def _create_component_with_shapes(self, component_name: str, shape_registry: Dict[str, tuple]) -> MettaModule:
        """Create a component instance with inferred shapes."""
        config_info = self.component_configs[component_name]
        component_class = config_info["class"]
        base_config = config_info["config"].copy()
        in_keys = config_info["in_keys"]
        out_keys = config_info["out_keys"]

        # Ensure all dependencies are initialized first
        for dep_name in self.dependencies[component_name]:
            if dep_name not in self.components:
                dep_component = self._create_component_with_shapes(dep_name, shape_registry)
                super().register_component(dep_name, dep_component, self.dependencies[dep_name])

        # Infer input shapes from shape registry
        inferred_shapes = {}
        for in_key in in_keys:
            if in_key not in shape_registry:
                raise ValueError(f"Cannot infer shape for key '{in_key}' needed by '{component_name}'")
            inferred_shapes[in_key] = shape_registry[in_key]

        # Update config with inferred shapes
        updated_config = self._update_config_with_shapes(
            component_class, base_config, inferred_shapes, in_keys, out_keys
        )

        # Create the component instance
        component = component_class(**updated_config)

        # Infer output shapes through dry run and update shape registry
        self._infer_output_shapes(component, shape_registry, inferred_shapes)

        return component

    def _update_config_with_shapes(
        self,
        component_class,
        config: Dict[str, Any],
        input_shapes: Dict[str, tuple],
        in_keys: List[str],
        out_keys: List[str],
    ) -> Dict[str, Any]:
        """Update component config with inferred input dimensions."""
        updated_config = config.copy()

        # Allow components to define custom shape inference
        if hasattr(component_class, "_update_config_from_shapes"):
            return component_class._update_config_from_shapes(config, input_shapes, in_keys, out_keys)

        # Default patterns for common component types
        if "LinearModule" in component_class.__name__:
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config["in_features"] = input_shape[-1]  # Last dimension
                updated_config["in_key"] = in_keys[0]
                updated_config["out_key"] = out_keys[0]

        elif "Conv2dModule" in component_class.__name__:
            if len(in_keys) == 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config["in_channels"] = input_shape[0]  # Assuming CHW format
                updated_config["in_key"] = in_keys[0]
                updated_config["out_key"] = out_keys[0]

        elif "LSTMModule" in component_class.__name__:
            if len(in_keys) >= 1:
                input_shape = input_shapes[in_keys[0]]
                updated_config["input_size"] = input_shape[-1]
                # LSTMModule uses different key names
                updated_config["in_key"] = in_keys[0]
                updated_config["out_key"] = out_keys[0]

        return updated_config

    def _infer_output_shapes(
        self, component: MettaModule, shape_registry: Dict[str, tuple], input_shapes: Dict[str, tuple]
    ):
        """Infer output shapes through a dry forward pass."""
        # Create mock input tensordict
        mock_inputs = {}
        for in_key, shape in input_shapes.items():
            mock_inputs[in_key] = torch.zeros(1, *shape)

        mock_tensordict = TensorDict(mock_inputs, batch_size=1)

        # Dry run to get output shapes
        with torch.no_grad():
            component.eval()
            result = component(mock_tensordict)

        # Record output shapes (remove batch dimension)
        for out_key in component.out_keys:
            if out_key in result:
                shape_registry[out_key] = result[out_key].shape[1:]

    def _topological_sort_configs(self):
        """Get component configs in dependency order."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(component_name):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            if component_name in visited:
                return

            temp_visited.add(component_name)
            for dep_name in self.dependencies.get(component_name, []):
                visit(dep_name)
            temp_visited.remove(component_name)
            visited.add(component_name)
            result.append(component_name)

        for component_name in self.component_configs.keys():
            if component_name not in visited:
                visit(component_name)

        return result

    def _update_container_keys_from_configs(self):
        """Update container's in_keys and out_keys based on component configurations."""
        all_in_keys = set()
        all_out_keys = set()

        for config_info in self.component_configs.values():
            all_in_keys.update(config_info["in_keys"])
            all_out_keys.update(config_info["out_keys"])

        # Container's inputs are keys not produced by any component
        self.in_keys = [key for key in all_in_keys if key not in all_out_keys]
        self.out_keys = list(all_out_keys)

    def __repr__(self) -> str:
        """Human-readable representation."""
        if self.initialized:
            lines = [f"LazyComponentContainer with {len(self.components)} components (initialized):"]
            for name, component in self.components.items():
                deps = self.dependencies.get(name, [])
                lines.append(f"  {name}: {type(component).__name__} (deps: {deps})")
        else:
            lines = [f"LazyComponentContainer with {len(self.component_configs)} component configs (not initialized):"]
            for name, config_info in self.component_configs.items():
                deps = self.dependencies.get(name, [])
                class_name = config_info["class"].__name__
                lines.append(f"  {name}: {class_name} (deps: {deps})")
        return "\n".join(lines)


class SafeLazyComponentContainer(LazyComponentContainer, SafeComponentContainer):
    """Combines safety validation with lazy initialization.

    This is the full-featured container that provides:
    - Deferred initialization with shape inference (from LazyComponentContainer)
    - Comprehensive validation (from SafeComponentContainer)
    - Elegant recursive execution (from base ComponentContainer)
    """

    def register_component_config(
        self,
        name: str,
        component_class,
        config: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ):
        """Register component configuration with validation."""
        # Validate the component class has proper interface
        if not hasattr(component_class, "__bases__") or not any(
            issubclass(base, MettaModule) for base in component_class.__mro__
        ):
            raise TypeError("Component class must inherit from MettaModule")

        # Use LazyComponentContainer's registration
        LazyComponentContainer.register_component_config(
            self, name, component_class, config, dependencies, in_keys, out_keys
        )

    def initialize_with_input_shapes(self, input_shapes: Dict[str, tuple]):
        """Initialize with both lazy initialization and safety validation."""
        # The safety validation happens automatically when LazyComponentContainer
        # calls super().register_component() during initialization, which triggers
        # SafeComponentContainer's validation logic
        LazyComponentContainer.initialize_with_input_shapes(self, input_shapes)
