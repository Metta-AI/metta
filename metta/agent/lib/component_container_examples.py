"""
Component Container Architecture Examples

This module demonstrates how to use the new wrapper-based ComponentContainer
architecture with its different layers: base, safe, lazy, and combined.
"""

import torch
from tensordict import TensorDict

from metta.agent.lib.component_container import (
    ComponentContainer,
    LazyComponentContainer,
    SafeComponentContainer,
    SafeLazyComponentContainer,
)
from metta.agent.lib.metta_moduly import LinearModule


def example_1_basic_container():
    """Example 1: Using the base ComponentContainer with immediate registration."""
    print("=" * 60)
    print("Example 1: Base ComponentContainer")
    print("=" * 60)

    # Create container
    container = ComponentContainer()

    # Create actual component instances (you need to know the dimensions)
    obs_processor = LinearModule(in_features=64, out_features=128, in_key="observation", out_key="features")
    policy = LinearModule(in_features=128, out_features=8, in_key="features", out_key="action")

    # Register components with dependencies
    container.register_component("obs_processor", obs_processor)
    container.register_component("policy", policy, dependencies=["obs_processor"])

    print(f"Container structure:\n{container}")
    print(f"Container inputs: {container.in_keys}")
    print(f"Container outputs: {container.out_keys}")

    # Use the container
    td = TensorDict({"observation": torch.randn(4, 64)}, batch_size=4)

    # Option 1: Execute specific component (with recursive dependencies)
    result = container.execute_component("policy", td)
    print(f"After executing 'policy': {list(result.keys())}")
    print(f"Action shape: {result['action'].shape}")

    # Option 2: Execute full network
    td_fresh = TensorDict({"observation": torch.randn(2, 64)}, batch_size=2)
    result = container.execute_network(td_fresh)
    print(f"After executing full network: {list(result.keys())}")


def example_2_safe_container():
    """Example 2: Using SafeComponentContainer with validation."""
    print("\n" + "=" * 60)
    print("Example 2: SafeComponentContainer (with validation)")
    print("=" * 60)

    # Create safe container
    container = SafeComponentContainer()

    # This will work fine
    obs_processor = LinearModule(in_features=64, out_features=128, in_key="observation", out_key="features")
    container.register_component("obs_processor", obs_processor)

    # This will catch conflicts
    try:
        # Try to register component with conflicting output key
        bad_component = LinearModule(
            in_features=32, out_features=64, in_key="other_input", out_key="features"
        )  # Same output key!
        container.register_component("bad_component", bad_component)
    except ValueError as e:
        print(f"Caught validation error: {e}")

    # This will catch missing dependencies
    try:
        policy = LinearModule(in_features=128, out_features=8, in_key="features", out_key="action")
        container.register_component("policy", policy, dependencies=["missing_component"])
    except ValueError as e:
        print(f"Caught dependency error: {e}")

    print("SafeComponentContainer provides comprehensive validation!")


def example_3_lazy_container():
    """Example 3: Using LazyComponentContainer with shape inference."""
    print("\n" + "=" * 60)
    print("Example 3: LazyComponentContainer (with shape inference)")
    print("=" * 60)

    # Create lazy container
    container = LazyComponentContainer()

    # Register component configurations (no actual instances yet)
    container.register_component_config(
        name="obs_processor",
        component_class=LinearModule,
        config={"out_features": 128},  # in_features will be inferred
        in_keys=["observation"],
        out_keys=["features"],
    )

    container.register_component_config(
        name="policy",
        component_class=LinearModule,
        config={"out_features": 8},  # in_features will be inferred from obs_processor output
        dependencies=["obs_processor"],
        in_keys=["features"],
        out_keys=["action"],
    )

    container.register_component_config(
        name="value",
        component_class=LinearModule,
        config={"out_features": 1},  # in_features will be inferred from obs_processor output
        dependencies=["obs_processor"],
        in_keys=["features"],
        out_keys=["value"],
    )

    print(f"Before initialization:\n{container}")
    print(f"Container inputs: {container.in_keys}")
    print(f"Container outputs: {container.out_keys}")

    # Initialize with input shapes - this triggers shape inference
    container.initialize_with_input_shapes({"observation": (64,)})

    print(f"\nAfter initialization:\n{container}")

    # Check the inferred shapes
    obs_proc = container.components["obs_processor"]
    policy = container.components["policy"]
    value = container.components["value"]

    print(f"obs_processor: Linear({obs_proc.linear.in_features} -> {obs_proc.linear.out_features})")
    print(f"policy: Linear({policy.linear.in_features} -> {policy.linear.out_features})")
    print(f"value: Linear({value.linear.in_features} -> {value.linear.out_features})")

    # Use the container
    td = TensorDict({"observation": torch.randn(4, 64)}, batch_size=4)
    result = container.execute_component("policy", td)
    print(f"Execution result keys: {list(result.keys())}")


def example_4_safe_lazy_container():
    """Example 4: Using SafeLazyComponentContainer with both features."""
    print("\n" + "=" * 60)
    print("Example 4: SafeLazyComponentContainer (validation + shape inference)")
    print("=" * 60)

    # Create combined container
    container = SafeLazyComponentContainer()

    # This will validate the component class
    try:

        class BadClass:
            pass

        container.register_component_config(
            name="bad",
            component_class=BadClass,  # Not a MettaModule!
            config={},
            in_keys=["input"],
            out_keys=["output"],
        )
    except TypeError as e:
        print(f"Caught class validation error: {e}")

    # Register valid configurations
    container.register_component_config(
        name="encoder",
        component_class=LinearModule,
        config={"out_features": 256},
        in_keys=["observation"],
        out_keys=["encoded"],
    )

    container.register_component_config(
        name="actor",
        component_class=LinearModule,
        config={"out_features": 8},
        dependencies=["encoder"],
        in_keys=["encoded"],
        out_keys=["action"],
    )

    container.register_component_config(
        name="critic",
        component_class=LinearModule,
        config={"out_features": 1},
        dependencies=["encoder"],
        in_keys=["encoded"],
        out_keys=["value"],
    )

    # Initialize with validation and shape inference
    container.initialize_with_input_shapes({"observation": (84,)})

    print("Successfully initialized with both validation and shape inference!")
    print(f"Final container:\n{container}")

    # Use the container
    td = TensorDict({"observation": torch.randn(4, 84)}, batch_size=4)
    result = container.execute_component("actor", td)
    print(f"Actor execution result keys: {list(result.keys())}")
    print(f"Action shape: {result['action'].shape}")


def example_5_comparison():
    """Example 5: Comparison of different approaches."""
    print("\n" + "=" * 60)
    print("Example 5: When to use which container?")
    print("=" * 60)

    print("1. ComponentContainer (Base):")
    print("   - Use for: Simple cases where you already have component instances")
    print("   - Benefits: Fastest, most direct")
    print("   - Example: Testing, simple experiments, known architectures")

    print("\n2. SafeComponentContainer:")
    print("   - Use for: When you need validation and error checking")
    print("   - Benefits: Catches interface mismatches, dependency errors")
    print("   - Example: Production code, team development, CI/CD")

    print("\n3. LazyComponentContainer:")
    print("   - Use for: Dynamic architectures with unknown input shapes")
    print("   - Benefits: Automatic shape inference, config-driven")
    print("   - Example: Research, hyperparameter sweeps, variable inputs")

    print("\n4. SafeLazyComponentContainer:")
    print("   - Use for: Production systems with dynamic architectures")
    print("   - Benefits: Both validation and shape inference")
    print("   - Example: Production ML systems, robust research frameworks")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_container()
    example_2_safe_container()
    example_3_lazy_container()
    example_4_safe_lazy_container()
    example_5_comparison()

    print("\n" + "=" * 60)
    print("Wrapper-based ComponentContainer architecture is ready to use!")
    print("=" * 60)
