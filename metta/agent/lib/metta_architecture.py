"""
⚠️  DEPRECATED: This file is obsolete and superseded by metta_architecture_refactored.py

This is an earlier version of the modular architecture that combines graph structure
and execution in a single MettaGraph class. The newer refactored version properly
separates these concerns into:

- MettaGraph: Graph structure only
- GraphExecutor: Execution and state management only
- RecreationManager: Dynamic recreation capabilities only
- ShapePropagator: Shape change propagation only
- MettaSystem: Coordinator providing unified interface

DO NOT USE THIS FILE - Use metta_architecture_refactored.py instead.

This file is kept only for historical reference showing the design evolution.
"""

from typing import Any, Dict, List, Optional

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
        self._recreation_callback = None  # Called when module is recreated
        self._preserve_state = True  # Whether to preserve state during recreation

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
        """
        Setup module with given input shapes.

        Args:
            input_shapes: List of input tensor shapes (without batch dimension)
        """
        self._in_tensor_shapes = input_shapes
        self._calculate_output_shape()
        self._initialize()
        self._ready = True

    def recreate_net(self, preserve_weights: bool = True, preserve_state: bool = None):
        """
        Recreate the neural network, optionally preserving weights and state.

        Args:
            preserve_weights: Whether to try to preserve compatible weights
            preserve_state: Whether to preserve module state (defaults to self._preserve_state)

        Returns:
            bool: True if recreation was successful
        """
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
        """
        Update module configuration and recreate network.

        Args:
            **new_params: New parameters to update
        """
        # Update parameters
        self._nn_params.update(new_params)

        # Recreate with new parameters
        self.recreate_net(preserve_weights=True)

    def resize_output(self, new_output_size: int, preserve_weights: bool = True):
        """
        Resize the output dimension of the module.

        Args:
            new_output_size: New output size
            preserve_weights: Whether to preserve compatible weights
        """
        if hasattr(self, "output_size"):
            self.output_size = new_output_size
        else:
            self._nn_params["output_size"] = new_output_size

        self.recreate_net(preserve_weights=preserve_weights)

    def _transfer_compatible_weights(self, old_net, new_net):
        """
        Transfer compatible weights from old network to new network.

        Args:
            old_net: Old network
            new_net: New network
        """
        try:
            # Get state dicts
            old_state = old_net.state_dict()
            new_state = new_net.state_dict()

            # Transfer compatible parameters
            for name, new_param in new_state.items():
                if name in old_state:
                    old_param = old_state[name]

                    # Transfer if shapes are compatible
                    if self._shapes_compatible(old_param.shape, new_param.shape):
                        # Handle different sized tensors by copying what fits
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
        """
        Get current module state for preservation during recreation.

        Returns:
            Current state or None if stateless
        """
        # Base implementation returns None (stateless)
        # Override in stateful modules like LSTM
        return None

    def set_current_state(self, state: Any):
        """
        Set module state after recreation.

        Args:
            state: State to restore
        """
        # Base implementation does nothing (stateless)
        # Override in stateful modules like LSTM
        pass

    def set_recreation_callback(self, callback):
        """
        Set callback to be called when module is recreated.

        Args:
            callback: Function(module, shape_changed) to call after recreation
        """
        self._recreation_callback = callback

    def _calculate_output_shape(self):
        """
        Calculate output shape based on input shapes.

        This method should be overridden by subclasses.
        """
        # Default implementation, override in subclasses
        self._out_tensor_shape = self._in_tensor_shapes[0].copy() if self._in_tensor_shapes else []

    def _initialize(self):
        """Initialize module parameters."""
        self._net = self._make_net()

    def _make_net(self):
        """
        Create the actual neural network.

        This method should be overridden by subclasses.
        """
        return nn.Identity()  # Default implementation returns identity

    def forward(self, inputs: List[torch.Tensor]):
        """
        Forward pass with explicit inputs.

        Args:
            inputs: List of input tensors from source modules

        Returns:
            Output tensor
        """
        # Default implementation for single-input modules
        if self._net is None:
            raise RuntimeError(f"Module {self._name} not initialized")
        return self._net(inputs[0])

    def get_new_state(self) -> Optional[Any]:
        """
        Get updated state after forward pass.

        Returns:
            New state or None if stateless
        """
        return None


class MettaGraph:
    """
    Manages the structure and execution of a neural network composed of MettaModules.

    Responsible for:
    - Module connections
    - Execution order
    - Input/output routing
    - State management
    - Dynamic recreation coordination
    """

    def __init__(self):
        self._modules: Dict[str, MettaModule] = {}
        self._connections: Dict[str, List[str]] = {}  # module_name -> list of source module names
        self._execution_order: List[str] = []
        self._states: Dict[str, Any] = {}  # module_name -> state
        self._recreation_in_progress = False  # Prevent recursive recreations

    @property
    def modules(self) -> Dict[str, MettaModule]:
        """Get the modules dictionary."""
        return self._modules

    @property
    def connections(self) -> Dict[str, List[str]]:
        """Get the connections dictionary."""
        return self._connections

    @property
    def execution_order(self) -> List[str]:
        """Get the execution order."""
        return self._execution_order

    @property
    def states(self) -> Dict[str, Any]:
        """Get the states dictionary."""
        return self._states

    def add_module(self, module: MettaModule):
        """
        Add a module to the graph.

        Args:
            module: MettaModule to add

        Returns:
            self for method chaining
        """
        self._modules[module.name] = module
        self._connections[module.name] = []
        return self

    def connect(self, source_name: str, target_name: str):
        """
        Connect source module to target module.

        Args:
            source_name: Name of source module
            target_name: Name of target module

        Returns:
            self for method chaining
        """
        if source_name not in self._modules:
            raise ValueError(f"Source module {source_name} not found")
        if target_name not in self._modules:
            raise ValueError(f"Target module {target_name} not found")

        self._connections[target_name].append(source_name)
        return self

    def setup(self):
        """
        Setup all modules in topological order.

        Calculates execution order and initializes each module.
        """
        self._execution_order = self._topological_sort()

        # Setup each module with its input shapes
        for module_name in self._execution_order:
            module = self._modules[module_name]

            # Get input shapes from source modules
            input_shapes = []
            for source_name in self._connections[module_name]:
                source_module = self._modules[source_name]
                if not source_module.ready:
                    raise RuntimeError(f"Source module {source_name} not ready")
                input_shapes.append(source_module.output_shape)

            # Setup module with input shapes
            module.setup(input_shapes)

    def _topological_sort(self) -> List[str]:
        """
        Sort modules in topological order.

        Returns:
            List of module names in execution order
        """
        visited = set()
        temp = set()
        order = []

        def visit(node):
            if node in temp:
                raise ValueError(f"Cycle detected in graph at node {node}")
            if node in visited:
                return

            temp.add(node)
            for neighbor in self._connections[node]:
                visit(neighbor)

            temp.remove(node)
            visited.add(node)
            order.append(node)

        for node in self._modules:
            if node not in visited:
                visit(node)

        return list(reversed(order))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire graph.

        Args:
            inputs: Dictionary of input tensors keyed by module name

        Returns:
            Dictionary of output tensors keyed by module name
        """
        # Store intermediate outputs
        outputs = inputs.copy()

        # Execute modules in topological order
        for module_name in self._execution_order:
            module = self._modules[module_name]

            # Collect inputs for this module
            module_inputs = [outputs[source_name] for source_name in self._connections[module_name]]

            # Execute module
            module_output = module.forward(module_inputs)

            # Update state if module is stateful
            new_state = module.get_new_state()
            if new_state is not None:
                self._states[module_name] = new_state

            # Store output
            outputs[module_name] = module_output

        return outputs

    def recreate_module(self, module_name: str, preserve_weights: bool = True, propagate_changes: bool = True) -> bool:
        """
        Recreate a specific module and handle downstream effects.

        Args:
            module_name: Name of module to recreate
            preserve_weights: Whether to preserve weights during recreation
            propagate_changes: Whether to propagate shape changes to downstream modules

        Returns:
            bool: True if recreation was successful
        """
        if self._recreation_in_progress:
            return False  # Prevent recursive recreation

        if module_name not in self._modules:
            raise ValueError(f"Module {module_name} not found")

        self._recreation_in_progress = True

        try:
            module = self._modules[module_name]
            old_output_shape = module.output_shape.copy()

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
                self._propagate_shape_changes(module_name)

            return success

        finally:
            self._recreation_in_progress = False

    def _propagate_shape_changes(self, changed_module_name: str):
        """
        Propagate shape changes from a module to its downstream dependencies.

        Args:
            changed_module_name: Name of module that changed shape
        """
        # Find all modules that depend on the changed module
        affected_modules = []
        for module_name, sources in self._connections.items():
            if changed_module_name in sources:
                affected_modules.append(module_name)

        # Recreate affected modules in topological order
        for module_name in affected_modules:
            if module_name in self._execution_order:
                module = self._modules[module_name]

                # Update input shapes
                input_shapes = []
                for source_name in self._connections[module_name]:
                    source_module = self._modules[source_name]
                    input_shapes.append(source_module.output_shape)

                # Re-setup the module with new input shapes
                module._in_tensor_shapes = input_shapes
                module.recreate_net(preserve_weights=True)

                # Recursively handle downstream changes
                self._propagate_shape_changes(module_name)

    def batch_recreate_modules(self, module_names: List[str], preserve_weights: bool = True):
        """
        Recreate multiple modules efficiently.

        Args:
            module_names: List of module names to recreate
            preserve_weights: Whether to preserve weights
        """
        if self._recreation_in_progress:
            return

        self._recreation_in_progress = True

        try:
            # Sort modules in execution order to minimize propagation
            sorted_modules = [name for name in self._execution_order if name in module_names]

            for module_name in sorted_modules:
                self._modules[module_name].recreate_net(preserve_weights=preserve_weights)

            # Single shape propagation pass at the end
            for module_name in sorted_modules:
                self._propagate_shape_changes(module_name)

        finally:
            self._recreation_in_progress = False

    def resize_module_output(self, module_name: str, new_output_size: int, preserve_weights: bool = True):
        """
        Resize a module's output and handle downstream effects.

        Args:
            module_name: Name of module to resize
            new_output_size: New output size
            preserve_weights: Whether to preserve weights
        """
        if module_name not in self._modules:
            raise ValueError(f"Module {module_name} not found")

        module = self._modules[module_name]
        module.resize_output(new_output_size, preserve_weights=preserve_weights)

        # Propagate changes
        self._propagate_shape_changes(module_name)

    def update_module_config(self, module_name: str, **new_params):
        """
        Update a module's configuration and recreate it.

        Args:
            module_name: Name of module to update
            **new_params: New configuration parameters
        """
        if module_name not in self._modules:
            raise ValueError(f"Module {module_name} not found")

        self._modules[module_name].update_config(**new_params)

        # Handle potential shape changes
        self._propagate_shape_changes(module_name)

    def get_recreation_status(self) -> Dict[str, Any]:
        """
        Get status information about recreation capabilities.

        Returns:
            Dict with recreation status information
        """
        return {
            "recreation_in_progress": self._recreation_in_progress,
            "modules": {name: module.ready for name, module in self._modules.items()},
            "connection_count": sum(len(sources) for sources in self._connections.values()),
        }


class LSTMModule(MettaModule):
    """
    LSTM module that handles tensor reshaping and state management automatically.
    """

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
        """
        Forward pass with state management.

        Args:
            inputs: List containing [input_tensor]

        Returns:
            LSTM output
        """
        if not inputs or self._net is None:
            raise RuntimeError("Invalid inputs or uninitialized module")

        x = inputs[0]
        state = self._get_state()

        # Handle the case where state is None
        if state is None:
            output, new_state = self._net(x)
        else:
            output, new_state = self._net(x, state)

        self._new_state = new_state

        return output

    def _get_state(self):
        """Get LSTM state."""
        return None  # This would be retrieved from MettaGraph

    def get_new_state(self):
        """Get updated state after forward pass."""
        return self._new_state

    def get_current_state(self):
        """Get current LSTM state for preservation during recreation."""
        return getattr(self, "_current_lstm_state", None)

    def set_current_state(self, state):
        """Set LSTM state after recreation."""
        if state is not None:
            self._current_lstm_state = state


class LinearModule(MettaModule):
    """
    Linear transformation module.
    """

    def __init__(self, name: str, output_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.output_size = output_size

    def _calculate_output_shape(self):
        """Calculate output shape based on input shapes."""
        self._out_tensor_shape = [self.output_size]

    def _make_net(self):
        """Create linear network."""
        return nn.Linear(self._in_tensor_shapes[0][0], self.output_size)


# Example usage
def example_usage():
    # Create modules
    input_processor = LinearModule("input_processor", output_size=64)
    lstm = LSTMModule("lstm", hidden_size=128)
    output_head = LinearModule("output_head", output_size=10)

    # Create graph
    graph = MettaGraph()
    graph.add_module(input_processor)
    graph.add_module(lstm)
    graph.add_module(output_head)

    # Connect modules
    graph.connect("input_processor", "lstm")
    graph.connect("lstm", "output_head")

    # Setup graph
    graph.setup()

    # Forward pass
    batch_size = 32
    seq_len = 10
    input_size = 20

    # Create random input tensor
    input_tensor = torch.rand(batch_size, seq_len, input_size)

    # Create inputs dictionary with the input tensor
    inputs = {"input_processor": input_tensor}

    # Run forward pass through the graph
    outputs = graph.forward(inputs)

    # Get the final output
    final_output = outputs["output_head"]

    return final_output


if __name__ == "__main__":
    # Run example
    output = example_usage()
    print("Output shape:", output.shape)
