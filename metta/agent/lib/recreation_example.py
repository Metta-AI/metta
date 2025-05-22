"""
Example demonstrating dynamic recreation capabilities in the Metta architecture.

This example shows how to:
1. Recreate individual modules while preserving weights
2. Resize modules dynamically during runtime
3. Update module configurations
4. Handle shape propagation across the graph
"""

import torch
from metta_architecture import LinearModule, LSTMModule, MettaGraph


def basic_recreation_example():
    """Basic example of module recreation."""
    print("=== Basic Recreation Example ===")

    # Create a simple network
    encoder = LinearModule("encoder", output_size=64)
    decoder = LinearModule("decoder", output_size=10)

    graph = MettaGraph()
    graph.add_module(encoder).add_module(decoder)
    graph.connect("encoder", "decoder")
    graph.setup()

    # Test forward pass
    inputs = {"encoder": torch.randn(32, 20)}
    outputs = graph.forward(inputs)
    print(f"Original output shape: {outputs['decoder'].shape}")

    # Recreate the encoder with preserved weights
    print("Recreating encoder...")
    success = graph.recreate_module("encoder", preserve_weights=True)
    print(f"Recreation successful: {success}")

    # Test forward pass again
    outputs = graph.forward(inputs)
    print(f"After recreation output shape: {outputs['decoder'].shape}")


def dynamic_resizing_example():
    """Example of dynamically resizing modules."""
    print("\n=== Dynamic Resizing Example ===")

    # Create network with LSTM
    encoder = LinearModule("encoder", output_size=32)
    lstm = LSTMModule("lstm", hidden_size=64)
    decoder = LinearModule("decoder", output_size=5)

    graph = MettaGraph()
    graph.add_module(encoder).add_module(lstm).add_module(decoder)
    graph.connect("encoder", "lstm")
    graph.connect("lstm", "decoder")
    graph.setup()

    # Initial forward pass
    inputs = {"encoder": torch.randn(16, 10, 20)}
    outputs = graph.forward(inputs)
    print(f"Initial decoder output shape: {outputs['decoder'].shape}")

    # Resize LSTM hidden size
    print("Resizing LSTM from 64 to 128...")
    graph.update_module_config("lstm", hidden_size=128)

    # Forward pass after resize
    outputs = graph.forward(inputs)
    print(f"After LSTM resize, decoder output shape: {outputs['decoder'].shape}")

    # Resize decoder output
    print("Resizing decoder output from 5 to 15...")
    graph.resize_module_output("decoder", new_output_size=15)

    # Final forward pass
    outputs = graph.forward(inputs)
    print(f"Final decoder output shape: {outputs['decoder'].shape}")


def curriculum_learning_example():
    """Example of gradual network growth for curriculum learning."""
    print("\n=== Curriculum Learning Example ===")

    # Start with small network
    encoder = LinearModule("encoder", output_size=16)
    core = LinearModule("core", output_size=32)
    decoder = LinearModule("decoder", output_size=4)

    graph = MettaGraph()
    graph.add_module(encoder).add_module(core).add_module(decoder)
    graph.connect("encoder", "core")
    graph.connect("core", "decoder")
    graph.setup()

    # Simulate training phases
    training_phases = [
        {"phase": 1, "encoder_size": 32, "core_size": 64},
        {"phase": 2, "encoder_size": 64, "core_size": 128},
        {"phase": 3, "encoder_size": 128, "core_size": 256},
    ]

    for phase_config in training_phases:
        print(f"\nPhase {phase_config['phase']} - Growing network capacity...")

        # Resize modules
        graph.resize_module_output("encoder", phase_config["encoder_size"])
        graph.resize_module_output("core", phase_config["core_size"])

        # Test forward pass
        inputs = {"encoder": torch.randn(8, 20)}
        outputs = graph.forward(inputs)

        print(f"  Encoder output: {graph.modules['encoder'].output_shape}")
        print(f"  Core output: {graph.modules['core'].output_shape}")
        print(f"  Final output shape: {outputs['decoder'].shape}")


def architecture_experimentation_example():
    """Example of experimenting with different architectures."""
    print("\n=== Architecture Experimentation Example ===")

    # Create base network
    encoder = LinearModule("encoder", output_size=64)
    processor = LinearModule("processor", output_size=64)
    decoder = LinearModule("decoder", output_size=10)

    graph = MettaGraph()
    graph.add_module(encoder).add_module(processor).add_module(decoder)
    graph.connect("encoder", "processor")
    graph.connect("processor", "decoder")
    graph.setup()

    # Test original architecture
    inputs = {"encoder": torch.randn(16, 32)}
    outputs1 = graph.forward(inputs)
    print(f"Linear processor output: {outputs1['decoder'].shape}")

    # Replace linear processor with LSTM
    print("Replacing linear processor with LSTM...")

    # Remove old processor and add LSTM
    old_processor = graph.modules.pop("processor")
    lstm_processor = LSTMModule("processor", hidden_size=64)
    graph.add_module(lstm_processor)

    # Need to re-setup after architecture change
    graph.setup()

    # Test new architecture
    outputs2 = graph.forward(inputs)
    print(f"LSTM processor output: {outputs2['decoder'].shape}")

    # Show that we can batch recreate multiple modules
    print("Batch recreating encoder and decoder...")
    graph.batch_recreate_modules(["encoder", "decoder"])

    outputs3 = graph.forward(inputs)
    print(f"After batch recreation: {outputs3['decoder'].shape}")


def error_handling_example():
    """Example of error handling during recreation."""
    print("\n=== Error Handling Example ===")

    encoder = LinearModule("encoder", output_size=32)
    decoder = LinearModule("decoder", output_size=10)

    graph = MettaGraph()
    graph.add_module(encoder).add_module(decoder)
    graph.connect("encoder", "decoder")
    graph.setup()

    # Test recreation status
    status = graph.get_recreation_status()
    print(f"Recreation status: {status}")

    # Try to recreate non-existent module
    try:
        graph.recreate_module("non_existent")
    except ValueError as e:
        print(f"Expected error: {e}")

    # Show successful recreation
    success = graph.recreate_module("encoder")
    print(f"Encoder recreation successful: {success}")

    # Show status after recreation
    status = graph.get_recreation_status()
    print(f"Status after recreation: {status}")


if __name__ == "__main__":
    basic_recreation_example()
    dynamic_resizing_example()
    curriculum_learning_example()
    architecture_experimentation_example()
    error_handling_example()

    print("\n=== All examples completed successfully! ===")
