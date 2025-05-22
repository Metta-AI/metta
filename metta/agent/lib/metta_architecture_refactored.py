"""
Refactored Metta Neural Network Architecture

This module defines a clean architecture for the Metta neural network system,
following the Single Responsibility Principle with proper separation of concerns.

Key components:
- MettaModule: Base class for all neural network modules
- MettaGraph: Graph structure only (modules and connections)
- GraphExecutor: Execution and state management
- RecreationManager: Dynamic recreation capabilities
- ShapePropagator: Shape change propagation
- MettaSystem: Coordinator that brings everything together
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


class MettaModule(nn.Module):
    """
    Base class for all neural network modules in the Metta system.

    Modules are responsible for:
    - Computation (forward pass)
    - Shape calculation
    - Parameter initialization
    - Dynamic recreation and adaptation

    They are NOT responsible for:
    - Network structure
    - Connections to other modules
    - Execution order
    """

    def __init__(self, name: str, **cfg):
        super().__init__()
        self._name = name
        self._ready = False
        self._nn_params = cfg.get("_nn_params", {})
        self._in_tensor_shapes = []
        self._out_tensor_shape = []
        self._net = None
        self._recreation_callback = None
        self._preserve_state = True

        # State management functions - provided by GraphExecutor during setup
        self._get_state_fn: Optional[Callable[[], Any]] = None
        self._set_state_fn: Optional[Callable[[Any], None]] = None

    @property
    def name(self) -> str:
        """Get the module name."""
        return self._name

    @property
    def ready(self) -> bool:
        """Check if the module is ready for execution."""
        return self._ready

    @property
    def output_shape(self):
        """Get the output tensor shape."""
        if not self._ready:
            raise RuntimeError(f"Module {self._name} not ready")
        return self._out_tensor_shape

    def setup(self, input_shapes: List[List[int]]):
        """Setup module with given input shapes."""
        self._in_tensor_shapes = input_shapes
        self._calculate_output_shape()
        self._initialize()
        self._ready = True

    def recreate_net(self, preserve_weights: bool = True, preserve_state: Optional[bool] = None):
        """Recreate the neural network, optionally preserving weights and state."""
        if not self._ready:
            raise RuntimeError(f"Cannot recreate {self._name}: module not initialized")

        if preserve_state is None:
            preserve_state = self._preserve_state

        # Preserve old state and weights
        old_net = self._net
        old_state = self.get_current_state() if preserve_state else None
        old_output_shape = self._out_tensor_shape.copy()

        try:
            # Recreate the network
            self._initialize()

            # Transfer compatible weights if requested
            if preserve_weights and old_net is not None:
                self._transfer_compatible_weights(old_net, self._net)

            # Restore state if requested
            if preserve_state and old_state is not None:
                self.set_current_state(old_state)

            # Check if output shape changed
            shape_changed = old_output_shape != self._out_tensor_shape

            # Notify callback if registered
            if self._recreation_callback is not None:
                self._recreation_callback(self, shape_changed)

            return True

        except Exception as e:
            # Restore old network on failure
            self._net = old_net
            self._out_tensor_shape = old_output_shape
            raise RuntimeError(f"Failed to recreate {self._name}: {e}")

    def update_config(self, **new_params):
        """Update module configuration and recreate network."""
        self._nn_params.update(new_params)
        self.recreate_net(preserve_weights=True)

    def resize_output(self, new_output_size: int, preserve_weights: bool = True):
        """Resize the output dimension of the module."""
        if hasattr(self, "output_size"):
            self.output_size = new_output_size
        else:
            self._nn_params["output_size"] = new_output_size

        self.recreate_net(preserve_weights=preserve_weights)

    def set_recreation_callback(self, callback: Optional[Callable]):
        """Set callback to be called when module is recreated."""
        self._recreation_callback = callback

    def _transfer_compatible_weights(self, old_net, new_net):
        """Transfer compatible weights from old network to new network."""
        try:
            old_state = old_net.state_dict()
            new_state = new_net.state_dict()

            for name, new_param in new_state.items():
                if name in old_state:
                    old_param = old_state[name]

                    if self._shapes_compatible(old_param.shape, new_param.shape):
                        min_shape = tuple(
                            min(old_s, new_s) for old_s, new_s in zip(old_param.shape, new_param.shape, strict=False)
                        )
                        slices = tuple(slice(0, s) for s in min_shape)

                        with torch.no_grad():
                            new_param[slices] = old_param[slices]

        except Exception:
            # Weight transfer failed, but that's okay - just use new weights
            pass

    def _shapes_compatible(self, old_shape, new_shape):
        """Check if two parameter shapes are compatible for weight transfer."""
        return len(old_shape) == len(new_shape) and all(
            old_s > 0 and new_s > 0 for old_s, new_s in zip(old_shape, new_shape, strict=False)
        )

    def get_current_state(self) -> Optional[Any]:
        """Get current module state for preservation during recreation."""
        return None

    def set_current_state(self, state: Any):
        """Set module state after recreation."""
        pass

    def _calculate_output_shape(self):
        """Calculate output shape based on input shapes."""
        self._out_tensor_shape = self._in_tensor_shapes[0].copy() if self._in_tensor_shapes else []

    def _initialize(self):
        """Initialize module parameters."""
        self._net = self._make_net()

    def _make_net(self):
        """Create the actual neural network."""
        return nn.Identity()

    def forward(self, inputs: List[torch.Tensor]):
        """Forward pass with explicit inputs."""
        if self._net is None:
            raise RuntimeError(f"Module {self._name} not initialized")
        return self._net(inputs[0])

    def get_new_state(self) -> Optional[Any]:
        """Get updated state after forward pass."""
        return None

    def set_state_provider(self, get_state_fn: Callable[[], Any], set_state_fn: Callable[[Any], None]):
        """Set state provider functions from GraphExecutor."""
        self._get_state_fn = get_state_fn
        self._set_state_fn = set_state_fn

    def _get_current_state(self) -> Optional[Any]:
        """Get current state from executor."""
        if self._get_state_fn is not None:
            return self._get_state_fn()
        return None

    def _update_state(self, new_state: Any):
        """Update state in executor."""
        if self._set_state_fn is not None:
            self._set_state_fn(new_state)


class MettaGraph:
    """
    Manages only the graph structure - modules and their connections.

    Responsible for:
    - Module storage
    - Connection management
    - Topological ordering
    """

    def __init__(self):
        self._modules: Dict[str, MettaModule] = {}
        self._connections: Dict[str, List[str]] = {}  # module_name -> list of source module names

    @property
    def modules(self) -> Dict[str, MettaModule]:
        """Get the modules dictionary."""
        return self._modules

    @property
    def connections(self) -> Dict[str, List[str]]:
        """Get the connections dictionary."""
        return self._connections

    def add_module(self, module: MettaModule):
        """Add a module to the graph."""
        self._modules[module.name] = module
        self._connections[module.name] = []
        return self

    def connect(self, source_name: str, target_name: str):
        """Connect source module to target module."""
        if source_name not in self._modules:
            raise ValueError(f"Source module {source_name} not found")
        if target_name not in self._modules:
            raise ValueError(f"Target module {target_name} not found")

        self._connections[target_name].append(source_name)
        return self

    def get_execution_order(self) -> List[str]:
        """Calculate topological execution order."""
        visited = set()
        temp = set()
        order = []

        def visit(node):
            if node in temp:
                raise ValueError(f"Cycle detected in graph at node {node}")
            if node in visited:
                return

            temp.add(node)
            # Visit dependencies (sources) first
            for source in self._connections[node]:
                visit(source)

            temp.remove(node)
            visited.add(node)
            order.append(node)

        for node in self._modules:
            if node not in visited:
                visit(node)

        # Don't reverse - dependencies are visited first, so order is correct
        return order

    def get_dependent_modules(self, module_name: str) -> List[str]:
        """Get modules that depend on the given module."""
        dependents = []
        for module, sources in self._connections.items():
            if module_name in sources:
                dependents.append(module)
        return dependents

    def remove_module(self, module_name: str):
        """Remove a module from the graph."""
        if module_name in self._modules:
            del self._modules[module_name]
            del self._connections[module_name]

            # Remove from other modules' connections
            for connections in self._connections.values():
                if module_name in connections:
                    connections.remove(module_name)


class GraphExecutor:
    """
    Handles execution and state management.

    Responsible for:
    - Module setup and initialization
    - Forward pass execution
    - State management
    """

    def __init__(self, graph: MettaGraph):
        self._graph = graph
        self._states: Dict[str, Any] = {}
        self._execution_order: List[str] = []
        self._setup_complete = False

    @property
    def states(self) -> Dict[str, Any]:
        """Get the states dictionary."""
        return self._states

    @property
    def execution_order(self) -> List[str]:
        """Get the execution order."""
        return self._execution_order

    def setup(self):
        """Setup all modules in topological order."""
        self._execution_order = self._graph.get_execution_order()

        # Setup each module with its input shapes
        for module_name in self._execution_order:
            module = self._graph.modules[module_name]

            # Create state getter/setter functions for this specific module
            def make_state_getter(name):
                return lambda: self._states.get(name)

            def make_state_setter(name):
                return lambda state: self._states.update({name: state}) if state is not None else None

            # Provide state management functions to the module
            module.set_state_provider(
                get_state_fn=make_state_getter(module_name), set_state_fn=make_state_setter(module_name)
            )

            # Get input shapes from source modules
            input_shapes = []
            for source_name in self._graph.connections[module_name]:
                source_module = self._graph.modules[source_name]
                if not source_module.ready:
                    raise RuntimeError(f"Source module {source_name} not ready")
                input_shapes.append(source_module.output_shape)

            # If no source modules, use pre-defined input shapes (for input modules)
            if not input_shapes and hasattr(module, "_in_tensor_shapes") and module._in_tensor_shapes:
                input_shapes = module._in_tensor_shapes

            # Setup module with input shapes
            module.setup(input_shapes)

        self._setup_complete = True

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute forward pass through the entire graph."""
        if not self._setup_complete:
            raise RuntimeError("Graph not setup. Call setup() first.")

        # Store intermediate outputs
        outputs = inputs.copy()

        # Execute modules in topological order
        for module_name in self._execution_order:
            module = self._graph.modules[module_name]

            # Collect inputs for this module
            module_inputs = []

            # If module has source modules, get their outputs
            if self._graph.connections[module_name]:
                module_inputs = [outputs[source_name] for source_name in self._graph.connections[module_name]]
            # If module has no source modules, check if it's an input module
            elif module_name in inputs:
                module_inputs = [inputs[module_name]]
            else:
                raise RuntimeError(f"Module {module_name} has no inputs and is not in external inputs")

            # Execute module
            module_output = module.forward(module_inputs)

            # Update state if module is stateful
            new_state = module.get_new_state()
            if new_state is not None:
                self._states[module_name] = new_state

            # Store output
            outputs[module_name] = module_output

        return outputs

    def reset_states(self):
        """Reset all module states."""
        self._states.clear()

    def get_state(self, module_name: str) -> Optional[Any]:
        """Get state for a specific module."""
        return self._states.get(module_name)

    def set_state(self, module_name: str, state: Any):
        """Set state for a specific module."""
        self._states[module_name] = state


class ShapePropagator:
    """
    Handles shape change propagation through the graph.

    Responsible for:
    - Finding affected modules when shapes change
    - Propagating shape changes downstream
    """

    def __init__(self, graph: MettaGraph):
        self._graph = graph

    def find_affected_modules(self, changed_module_name: str) -> List[str]:
        """Find all modules affected by a shape change."""
        affected = []
        visited = set()

        def collect_dependents(module_name):
            if module_name in visited:
                return
            visited.add(module_name)

            dependents = self._graph.get_dependent_modules(module_name)
            for dependent in dependents:
                affected.append(dependent)
                collect_dependents(dependent)  # Recursively collect

        collect_dependents(changed_module_name)
        return affected

    def propagate_shape_changes(self, changed_module_name: str, executor: GraphExecutor):
        """Propagate shape changes from a module to its downstream dependencies."""
        affected_modules = self.find_affected_modules(changed_module_name)

        # Update affected modules in execution order
        execution_order = executor.execution_order
        for module_name in execution_order:
            if module_name in affected_modules:
                module = self._graph.modules[module_name]

                # Update input shapes
                input_shapes = []
                for source_name in self._graph.connections[module_name]:
                    source_module = self._graph.modules[source_name]
                    input_shapes.append(source_module.output_shape)

                # Re-setup the module with new input shapes
                module._in_tensor_shapes = input_shapes
                module.recreate_net(preserve_weights=True)


class RecreationManager:
    """
    Handles dynamic recreation of modules.

    Responsible for:
    - Module recreation coordination
    - Batch recreation operations
    - Recreation status tracking
    """

    def __init__(self, graph: MettaGraph, executor: GraphExecutor):
        self._graph = graph
        self._executor = executor
        self._shape_propagator = ShapePropagator(graph)
        self._recreation_in_progress = False

    def recreate_module(self, module_name: str, preserve_weights: bool = True, propagate_changes: bool = True) -> bool:
        """Recreate a specific module and handle downstream effects."""
        if self._recreation_in_progress:
            return False  # Prevent recursive recreation

        if module_name not in self._graph.modules:
            raise ValueError(f"Module {module_name} not found")

        self._recreation_in_progress = True

        try:
            module = self._graph.modules[module_name]

            # Set up callback to handle shape changes
            shape_changed = False

            def recreation_callback(recreated_module, did_shape_change):
                nonlocal shape_changed
                shape_changed = did_shape_change

            module.set_recreation_callback(recreation_callback)

            # Recreate the module
            success = module.recreate_net(preserve_weights=preserve_weights)

            if success and shape_changed and propagate_changes:
                # Propagate shape changes to downstream modules
                self._shape_propagator.propagate_shape_changes(module_name, self._executor)

            return success

        finally:
            self._recreation_in_progress = False

    def batch_recreate_modules(self, module_names: List[str], preserve_weights: bool = True):
        """Recreate multiple modules efficiently."""
        if self._recreation_in_progress:
            return

        self._recreation_in_progress = True

        try:
            # Sort modules in execution order to minimize propagation
            execution_order = self._executor.execution_order
            sorted_modules = [name for name in execution_order if name in module_names]

            for module_name in sorted_modules:
                self._graph.modules[module_name].recreate_net(preserve_weights=preserve_weights)

            # Single shape propagation pass at the end
            for module_name in sorted_modules:
                self._shape_propagator.propagate_shape_changes(module_name, self._executor)

        finally:
            self._recreation_in_progress = False

    def resize_module_output(self, module_name: str, new_output_size: int, preserve_weights: bool = True):
        """Resize a module's output and handle downstream effects."""
        if module_name not in self._graph.modules:
            raise ValueError(f"Module {module_name} not found")

        module = self._graph.modules[module_name]
        module.resize_output(new_output_size, preserve_weights=preserve_weights)

        # Propagate changes
        self._shape_propagator.propagate_shape_changes(module_name, self._executor)

    def update_module_config(self, module_name: str, **new_params):
        """Update a module's configuration and recreate it."""
        if module_name not in self._graph.modules:
            raise ValueError(f"Module {module_name} not found")

        self._graph.modules[module_name].update_config(**new_params)

        # Handle potential shape changes
        self._shape_propagator.propagate_shape_changes(module_name, self._executor)

    def get_recreation_status(self) -> Dict[str, Any]:
        """Get status information about recreation capabilities."""
        return {
            "recreation_in_progress": self._recreation_in_progress,
            "modules": {name: module.ready for name, module in self._graph.modules.items()},
            "connection_count": sum(len(sources) for sources in self._graph.connections.values()),
        }


class MettaSystem:
    """
    Coordinator that brings all components together.

    Provides a unified interface while delegating to appropriate components.
    """

    def __init__(self):
        self.graph = MettaGraph()
        self.executor = GraphExecutor(self.graph)
        self.recreation_manager = RecreationManager(self.graph, self.executor)

    # Graph structure delegation
    def add_module(self, module: MettaModule):
        """Add a module to the system."""
        return self.graph.add_module(module)

    def connect(self, source_name: str, target_name: str):
        """Connect two modules."""
        return self.graph.connect(source_name, target_name)

    def remove_module(self, module_name: str):
        """Remove a module from the system."""
        return self.graph.remove_module(module_name)

    # Execution delegation
    def setup(self):
        """Setup the entire system."""
        return self.executor.setup()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute forward pass."""
        return self.executor.forward(inputs)

    def reset_states(self):
        """Reset all module states."""
        return self.executor.reset_states()

    # Recreation delegation
    def recreate_module(self, module_name: str, preserve_weights: bool = True):
        """Recreate a module."""
        return self.recreation_manager.recreate_module(module_name, preserve_weights)

    def batch_recreate_modules(self, module_names: List[str], preserve_weights: bool = True):
        """Recreate multiple modules."""
        return self.recreation_manager.batch_recreate_modules(module_names, preserve_weights)

    def resize_module_output(self, module_name: str, new_output_size: int):
        """Resize a module's output."""
        return self.recreation_manager.resize_module_output(module_name, new_output_size)

    def update_module_config(self, module_name: str, **new_params):
        """Update module configuration."""
        return self.recreation_manager.update_module_config(module_name, **new_params)

    def get_recreation_status(self):
        """Get recreation status."""
        return self.recreation_manager.get_recreation_status()

    # Convenience properties
    @property
    def modules(self):
        """Access to modules."""
        return self.graph.modules

    @property
    def connections(self):
        """Access to connections."""
        return self.graph.connections

    @property
    def execution_order(self):
        """Access to execution order."""
        return self.executor.execution_order

    @property
    def states(self):
        """Access to module states."""
        return self.executor.states


# Specialized modules remain the same
class LSTMModule(MettaModule):
    """LSTM module that handles tensor reshaping and state management automatically."""

    def __init__(self, name: str, hidden_size: int, num_layers: int = 1, **kwargs):
        super().__init__(name, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._new_state = None

    def _calculate_output_shape(self):
        """Calculate output shape based on input shapes."""
        self._out_tensor_shape = [self.hidden_size]

    def _make_net(self):
        """Create LSTM network."""
        return nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, num_layers=self.num_layers, **self._nn_params)

    def forward(self, inputs: List[torch.Tensor]):
        """Forward pass with proper state management."""
        if not inputs or self._net is None:
            raise RuntimeError("Invalid inputs or uninitialized module")

        x = inputs[0]

        # Get current state from GraphExecutor via provided function
        current_state = self._get_current_state()

        # Forward pass through LSTM
        if current_state is None:
            output, new_state = self._net(x)
        else:
            output, new_state = self._net(x, current_state)

        # Store new state for GraphExecutor to pick up
        self._new_state = new_state
        return output

    def get_new_state(self):
        """Get updated state after forward pass."""
        return self._new_state

    def get_current_state(self):
        """Get current LSTM state for preservation during recreation."""
        return self._get_current_state()

    def set_current_state(self, state):
        """Set LSTM state after recreation."""
        if state is not None:
            self._update_state(state)


class LinearModule(MettaModule):
    """Linear transformation module."""

    def __init__(self, name: str, output_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.output_size = output_size

    def _calculate_output_shape(self):
        """Calculate output shape based on input shapes."""
        self._out_tensor_shape = [self.output_size]

    def _make_net(self):
        """Create linear network."""
        return nn.Linear(self._in_tensor_shapes[0][0], self.output_size)


# Example usage function
def example_usage():
    """Example of using the refactored architecture."""
    # Create system
    system = MettaSystem()

    # Create modules
    encoder = LinearModule("encoder", output_size=64)
    lstm = LSTMModule("lstm", hidden_size=128)
    decoder = LinearModule("decoder", output_size=10)

    # Add modules and connections
    system.add_module(encoder)
    system.add_module(lstm)
    system.add_module(decoder)

    system.connect("encoder", "lstm")
    system.connect("lstm", "decoder")

    # Setup system
    system.setup()

    # Forward pass
    inputs = {"encoder": torch.rand(32, 20)}
    outputs = system.forward(inputs)

    # Dynamic recreation
    system.recreate_module("encoder", preserve_weights=True)
    system.resize_module_output("decoder", new_output_size=20)

    return outputs


if __name__ == "__main__":
    output = example_usage()
    print("Refactored architecture example completed!")
    print(f"Output shape: {output['decoder'].shape}")
