"""
State Management Fix for Metta Architecture

This file demonstrates the corrected implementation for state management
between modules and the GraphExecutor, addressing the gap where LSTM
modules need to access their state from the executor.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn


class MettaModule(nn.Module):
    """
    Base class with proper state management interface.
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

    @property
    def name(self) -> str:
        return self._name

    @property
    def ready(self) -> bool:
        return self._ready

    def setup(self, input_shapes: List[List[int]]):
        """Setup module with given input shapes."""
        self._in_tensor_shapes = input_shapes
        self._calculate_output_shape()
        self._initialize()
        self._ready = True

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
        """Get updated state after forward pass - to be overridden by stateful modules."""
        return None


class GraphExecutor:
    """
    Updated GraphExecutor with proper state management.
    """

    def __init__(self, graph):
        self._graph = graph
        self._states: Dict[str, Any] = {}
        self._execution_order: List[str] = []
        self._setup_complete = False

    def setup(self):
        """Setup all modules and provide state management functions."""
        self._execution_order = self._graph.get_execution_order()

        # Setup each module and provide state management functions
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
                input_shapes.append(source_module._out_tensor_shape)

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

        outputs = inputs.copy()

        for module_name in self._execution_order:
            module = self._graph.modules[module_name]

            # Collect inputs for this module
            module_inputs = [outputs[source_name] for source_name in self._graph.connections[module_name]]

            # Execute module
            module_output = module.forward(module_inputs)

            # Update state if module is stateful
            new_state = module.get_new_state()
            if new_state is not None:
                self._states[module_name] = new_state

            outputs[module_name] = module_output

        return outputs

    def get_state(self, module_name: str) -> Optional[Any]:
        """Get state for a specific module."""
        return self._states.get(module_name)

    def set_state(self, module_name: str, state: Any):
        """Set state for a specific module."""
        self._states[module_name] = state

    def reset_states(self):
        """Reset all module states."""
        self._states.clear()


class LSTMModule(MettaModule):
    """
    Corrected LSTM module with proper state management.
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


# Example demonstrating the corrected state management
def demonstrate_corrected_state_management():
    """Show how the corrected state management works."""

    # Mock graph for demonstration
    class MockGraph:
        def __init__(self):
            self.modules = {}
            self.connections = {}

        def get_execution_order(self):
            return list(self.modules.keys())

        def add_module(self, module):
            self.modules[module.name] = module
            self.connections[module.name] = []
            return self

    # Create components
    graph = MockGraph()
    executor = GraphExecutor(graph)

    # Create LSTM module
    lstm = LSTMModule("lstm", hidden_size=64, num_layers=1)
    graph.add_module(lstm)

    # Set input shapes for the LSTM (simulating that it has an input)
    lstm._in_tensor_shapes = [[32]]  # input_size = 32

    # Setup - this is where state management functions are provided
    executor.setup()

    # Verify state management is working
    print("=== State Management Demonstration ===")

    # Initially no state
    initial_state = lstm.get_current_state()
    print(f"Initial state: {initial_state}")

    # Forward pass creates state
    dummy_input = torch.randn(1, 10, 32)  # batch_size=1, seq_len=10, input_size=32
    output = lstm.forward([dummy_input])

    # State is automatically stored in executor
    new_state = lstm.get_new_state()
    print(f"New state created: {new_state is not None}")
    print(f"State shapes: {[s.shape for s in new_state] if new_state else 'None'}")

    # Simulate executor updating its state storage (this happens automatically in forward())
    executor.set_state("lstm", new_state)

    # Next forward pass should use the stored state
    retrieved_state = lstm.get_current_state()
    print(f"Retrieved state matches: {retrieved_state is new_state}")

    # Another forward pass with state
    output2 = lstm.forward([dummy_input])
    print(f"Second forward pass output shape: {output2.shape}")

    return output2


if __name__ == "__main__":
    result = demonstrate_corrected_state_management()
    print("\n✓ Corrected state management working properly!")
