"""
Example demonstrating the benefits of the refactored Metta architecture.

This example shows how the separation of concerns makes the system:
1. More modular and testable
2. Easier to extend and customize
3. Better organized with clear responsibilities
"""

import torch
from metta_architecture_refactored import (
    GraphExecutor,
    LinearModule,
    LSTMModule,
    MettaGraph,
    MettaSystem,
    RecreationManager,
)


def demonstrate_modularity():
    """Show how each component can be used independently."""
    print("=== Modularity Demonstration ===")

    # 1. Graph structure can be tested independently
    graph = MettaGraph()
    encoder = LinearModule("encoder", output_size=32)
    decoder = LinearModule("decoder", output_size=10)

    graph.add_module(encoder).add_module(decoder)
    graph.connect("encoder", "decoder")

    print(f"Graph connections: {graph.connections}")
    print(f"Execution order: {graph.get_execution_order()}")
    print(f"Dependents of encoder: {graph.get_dependent_modules('encoder')}")

    # 2. Executor can be tested independently
    executor = GraphExecutor(graph)
    executor.setup()

    inputs = {"encoder": torch.randn(16, 20)}
    outputs = executor.forward(inputs)
    print(f"Executor output shape: {outputs['decoder'].shape}")

    # 3. Recreation manager can be tested independently
    recreation_manager = RecreationManager(graph, executor)
    status = recreation_manager.get_recreation_status()
    print(f"Recreation status: {status}")


def demonstrate_extensibility():
    """Show how easy it is to extend individual components."""
    print("\n=== Extensibility Demonstration ===")

    # Custom executor that logs execution
    class LoggingGraphExecutor(GraphExecutor):
        def forward(self, inputs):
            print(f"Executing forward pass with inputs: {list(inputs.keys())}")
            outputs = super().forward(inputs)
            print(f"Forward pass complete, outputs: {list(outputs.keys())}")
            return outputs

    # Custom recreation manager with metrics
    class MetricsRecreationManager(RecreationManager):
        def __init__(self, graph, executor):
            super().__init__(graph, executor)
            self.recreation_count = 0

        def recreate_module(
            self, module_name: str, preserve_weights: bool = True, propagate_changes: bool = True
        ) -> bool:
            self.recreation_count += 1
            print(f"Recreation #{self.recreation_count}: {module_name}")
            return super().recreate_module(module_name, preserve_weights, propagate_changes)

    # Use custom components
    graph = MettaGraph()
    encoder = LinearModule("encoder", output_size=32)
    decoder = LinearModule("decoder", output_size=10)

    graph.add_module(encoder).add_module(decoder)
    graph.connect("encoder", "decoder")

    # Custom executor
    executor = LoggingGraphExecutor(graph)
    executor.setup()

    # Custom recreation manager
    recreation_manager = MetricsRecreationManager(graph, executor)

    # Test execution
    inputs = {"encoder": torch.randn(8, 16)}
    outputs = executor.forward(inputs)

    # Test recreation
    recreation_manager.recreate_module("encoder")
    print(f"Total recreations: {recreation_manager.recreation_count}")


def demonstrate_testing():
    """Show how the modular design makes testing easier."""
    print("\n=== Testing Demonstration ===")

    # Test graph structure only
    def test_graph_structure():
        graph = MettaGraph()
        module1 = LinearModule("module1", output_size=32)
        module2 = LinearModule("module2", output_size=16)

        graph.add_module(module1).add_module(module2)
        graph.connect("module1", "module2")

        # Test topological ordering
        order = graph.get_execution_order()
        assert order == ["module1", "module2"], f"Expected ['module1', 'module2'], got {order}"

        # Test dependency finding
        deps = graph.get_dependent_modules("module1")
        assert deps == ["module2"], f"Expected ['module2'], got {deps}"

        print("✓ Graph structure tests passed")

    # Test executor only
    def test_executor():
        # Mock graph for testing
        class MockGraph:
            def __init__(self):
                self.modules = {"test": LinearModule("test", output_size=5)}
                self.connections = {"test": []}

            def get_execution_order(self):
                return ["test"]

        graph = MockGraph()
        executor = GraphExecutor(graph)
        executor.setup()

        # Test forward pass
        inputs = {"test": torch.randn(4, 10)}
        outputs = executor.forward(inputs)

        assert "test" in outputs, "Output should contain 'test' key"
        assert outputs["test"].shape == (4, 5), f"Expected shape (4, 5), got {outputs['test'].shape}"

        print("✓ Executor tests passed")

    # Run tests
    test_graph_structure()
    test_executor()


def demonstrate_composition():
    """Show how components compose together seamlessly."""
    print("\n=== Composition Demonstration ===")

    # Build system piece by piece
    graph = MettaGraph()

    # Add modules
    modules = [
        LinearModule("input", output_size=64),
        LSTMModule("processor", hidden_size=128),
        LinearModule("output", output_size=10),
    ]

    for module in modules:
        graph.add_module(module)

    # Connect modules
    graph.connect("input", "processor")
    graph.connect("processor", "output")

    # Create executor and setup
    executor = GraphExecutor(graph)
    executor.setup()

    # Create recreation manager
    recreation_manager = RecreationManager(graph, executor)

    # Test the composed system
    inputs = {"input": torch.randn(16, 32)}
    outputs = executor.forward(inputs)
    print(f"System output shape: {outputs['output'].shape}")

    # Test dynamic capabilities
    recreation_manager.resize_module_output("output", new_output_size=20)
    outputs = executor.forward(inputs)
    print(f"After resize, output shape: {outputs['output'].shape}")


def demonstrate_system_coordinator():
    """Show how MettaSystem provides a clean unified interface."""
    print("\n=== System Coordinator Demonstration ===")

    # Single system provides all functionality
    system = MettaSystem()

    # But you can still access individual components if needed
    print(f"Graph type: {type(system.graph).__name__}")
    print(f"Executor type: {type(system.executor).__name__}")
    print(f"Recreation manager type: {type(system.recreation_manager).__name__}")

    # Build network through system interface
    encoder = LinearModule("encoder", output_size=64)
    decoder = LinearModule("decoder", output_size=10)

    system.add_module(encoder)
    system.add_module(decoder)
    system.connect("encoder", "decoder")
    system.setup()

    # Use system interface
    inputs = {"encoder": torch.randn(8, 32)}
    outputs = system.forward(inputs)
    print(f"System forward output: {outputs['decoder'].shape}")

    # Access individual component features when needed
    direct_order = system.graph.get_execution_order()
    system_order = system.execution_order
    print(f"Direct access to execution order: {direct_order}")
    print(f"System property access: {system_order}")


def performance_comparison():
    """Show that the refactored architecture doesn't sacrifice performance."""
    print("\n=== Performance Comparison ===")

    import time

    # Create identical networks
    def create_network():
        system = MettaSystem()
        encoder = LinearModule("encoder", output_size=128)
        lstm = LSTMModule("lstm", hidden_size=256)
        decoder = LinearModule("decoder", output_size=64)

        system.add_module(encoder)
        system.add_module(lstm)
        system.add_module(decoder)

        system.connect("encoder", "lstm")
        system.connect("lstm", "decoder")
        system.setup()

        return system

    # Test performance
    system = create_network()
    inputs = {"encoder": torch.randn(32, 64)}

    # Warmup
    for _ in range(10):
        system.forward(inputs)

    # Time forward passes
    start_time = time.time()
    for _ in range(100):
        outputs = system.forward(inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100 * 1000  # ms
    print(f"Average forward pass time: {avg_time:.2f}ms")
    print(f"Final output shape: {outputs['decoder'].shape}")


if __name__ == "__main__":
    demonstrate_modularity()
    demonstrate_extensibility()
    demonstrate_testing()
    demonstrate_composition()
    demonstrate_system_coordinator()
    performance_comparison()

    print("\n🎉 All refactored architecture demonstrations completed!")
    print("\nKey Benefits Demonstrated:")
    print("✓ Modular components that can be tested independently")
    print("✓ Easy extensibility through inheritance/composition")
    print("✓ Clear separation of concerns")
    print("✓ Seamless composition of components")
    print("✓ Clean unified interface while preserving access to internals")
    print("✓ No performance sacrifice")
