import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.component_container import (
    ComponentContainer,
    LazyComponentContainer,
    SafeComponentContainer,
    SafeLazyComponentContainer,
)
from metta.agent.lib.metta_modules import LinearModule


class TestBaseComponentContainer:
    """Test the base ComponentContainer with immediate registration."""

    def test_basic_registration(self):
        """Test basic component registration with actual instances."""
        container = ComponentContainer()

        # Create actual component instance
        component = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")

        container.register_component("test_module", component)

        assert "test_module" in container.components
        assert container.get_component_dependencies("test_module") == []
        assert container.in_keys == ["input"]
        assert container.out_keys == ["output"]

    def test_dependency_registration(self):
        """Test component registration with dependencies."""
        container = ComponentContainer()

        # Create actual component instances
        obs_processor = LinearModule(in_features=10, out_features=8, in_key="observation", out_key="processed_obs")
        policy = LinearModule(in_features=8, out_features=3, in_key="processed_obs", out_key="action")

        container.register_component("obs_processor", obs_processor)
        container.register_component("policy", policy, dependencies=["obs_processor"])

        assert container.get_component_dependencies("obs_processor") == []
        assert container.get_component_dependencies("policy") == ["obs_processor"]
        assert container.in_keys == ["observation"]
        assert set(container.out_keys) == {"processed_obs", "action"}

    def test_execution_order_calculation(self):
        """Test execution order computation for dependency chains."""
        container = ComponentContainer()

        # Create dependency chain: A -> B -> C
        comp_a = LinearModule(in_features=10, out_features=8, in_key="input", out_key="a_out")
        comp_b = LinearModule(in_features=8, out_features=6, in_key="a_out", out_key="b_out")
        comp_c = LinearModule(in_features=6, out_features=3, in_key="b_out", out_key="output")

        container.register_component("A", comp_a)
        container.register_component("B", comp_b, dependencies=["A"])
        container.register_component("C", comp_c, dependencies=["B"])

        order = container.get_execution_order("C")
        assert order == ["A", "B", "C"]

        order = container.get_execution_order("B")
        assert order == ["A", "B"]

    def test_recursive_execution(self):
        """Test recursive execution of dependencies."""
        container = ComponentContainer()

        # Create actual component instances
        obs_processor = LinearModule(in_features=10, out_features=8, in_key="observation", out_key="processed_obs")
        policy = LinearModule(in_features=8, out_features=3, in_key="processed_obs", out_key="action")

        container.register_component("obs_processor", obs_processor)
        container.register_component("policy", policy, dependencies=["obs_processor"])

        # Test data
        td = TensorDict({"observation": torch.randn(2, 10)}, batch_size=2)

        # Execute policy (should recursively execute obs_processor)
        result = container.execute_component("policy", td)

        # Both outputs should be present
        assert "processed_obs" in result
        assert "action" in result
        assert result["processed_obs"].shape == (2, 8)
        assert result["action"].shape == (2, 3)

    def test_output_presence_check(self):
        """Test that components skip execution if outputs already exist."""
        container = ComponentContainer()

        component = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
        container.register_component("test", component)

        # Pre-populate output
        td = TensorDict({"input": torch.randn(2, 10), "output": torch.randn(2, 5)}, batch_size=2)
        original_output = td["output"].clone()

        result = container.execute_component("test", td)

        # Should return original pre-populated output unchanged
        assert torch.equal(result["output"], original_output)

    def test_hotswapping(self):
        """Test component replacement (hotswapping)."""
        container = ComponentContainer()

        # Register original component
        original = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
        container.register_component("swappable", original)

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        # Execute with original
        result1 = container.execute_component("swappable", td)
        assert result1["output"].shape == (2, 5)

        # Replace with different size component
        replacement = LinearModule(in_features=10, out_features=8, in_key="input", out_key="output")
        container.replace_component("swappable", replacement)

        # Use fresh tensordict to ensure no cached outputs
        td_fresh = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
        result2 = container.execute_component("swappable", td_fresh)
        assert result2["output"].shape == (2, 8)

    def test_execute_network(self):
        """Test full network execution."""
        container = ComponentContainer()

        # Create actual component instances
        obs_processor = LinearModule(in_features=10, out_features=8, in_key="observation", out_key="processed_obs")
        policy = LinearModule(in_features=8, out_features=3, in_key="processed_obs", out_key="action")

        container.register_component("obs_processor", obs_processor)
        container.register_component("policy", policy, dependencies=["obs_processor"])

        # Test data
        td = TensorDict({"observation": torch.randn(2, 10)}, batch_size=2)

        # Execute full network
        result = container.execute_network(td)

        # All outputs should be present
        assert "processed_obs" in result
        assert "action" in result
        assert result["processed_obs"].shape == (2, 8)
        assert result["action"].shape == (2, 3)

    def test_repr(self):
        """Test string representation."""
        container = ComponentContainer()

        comp_a = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")
        comp_b = LinearModule(in_features=5, out_features=3, in_key="output", out_key="final")

        container.register_component("A", comp_a)
        container.register_component("B", comp_b, dependencies=["A"])

        repr_str = repr(container)
        assert "ComponentContainer with 2 components" in repr_str
        assert "A: LinearModule (deps: [])" in repr_str
        assert "B: LinearModule (deps: ['A'])" in repr_str


class TestSafeComponentContainer:
    """Test the SafeComponentContainer with validation."""

    def test_invalid_component_type(self):
        """Test validation catches non-MettaModule components."""
        container = SafeComponentContainer()

        with pytest.raises(TypeError, match="must be a MettaModule"):
            container.register_component("bad", torch.nn.Linear(10, 5))

    def test_missing_attributes(self):
        """Test validation catches components without required attributes."""
        container = SafeComponentContainer()

        # Create a mock component without proper attributes
        class BadComponent:
            pass

        bad_component = BadComponent()

        with pytest.raises(TypeError, match="must be a MettaModule"):
            container.register_component("bad", bad_component)

    def test_invalid_attribute_types(self):
        """Test validation catches components with wrong attribute types."""
        container = SafeComponentContainer()

        # Create a mock component with wrong attribute types
        class BadComponent:
            def __init__(self):
                self.in_keys = "not_a_list"  # Should be a list
                self.out_keys = "not_a_list"  # Should be a list

        bad_component = BadComponent()

        with pytest.raises(TypeError, match="must be a MettaModule"):
            container.register_component("bad", bad_component)

    def test_output_key_conflicts(self):
        """Test validation catches output key conflicts."""
        container = SafeComponentContainer()

        # Register first component
        comp1 = LinearModule(in_features=10, out_features=5, in_key="input1", out_key="conflict")
        container.register_component("comp1", comp1)

        # Try to register second component with conflicting output key
        comp2 = LinearModule(in_features=10, out_features=5, in_key="input2", out_key="conflict")

        with pytest.raises(ValueError, match="Output key conflict"):
            container.register_component("comp2", comp2)

    def test_missing_dependency(self):
        """Test validation catches missing dependencies."""
        container = SafeComponentContainer()

        comp = LinearModule(in_features=10, out_features=5, in_key="input", out_key="output")

        with pytest.raises(ValueError, match="Dependency 'missing' not found"):
            container.register_component("comp", comp, dependencies=["missing"])

    def test_circular_dependencies(self):
        """Test validation catches circular dependencies."""
        container = SafeComponentContainer()

        # Create components that would form a circular dependency
        comp_a = LinearModule(in_features=10, out_features=8, in_key="input_a", out_key="output_a")
        comp_b = LinearModule(in_features=8, out_features=6, in_key="input_b", out_key="output_b")
        comp_c = LinearModule(in_features=6, out_features=10, in_key="input_c", out_key="output_c")

        # Register components to create circular dependency: A -> B -> C -> A
        container.register_component("A", comp_a)
        container.register_component("B", comp_b, dependencies=["A"])

        with pytest.raises(ValueError, match="Circular dependency detected"):
            container.register_component("C", comp_c, dependencies=["B"])
            # This would create a cycle if we tried: A depends on C
            comp_a_modified = LinearModule(in_features=6, out_features=8, in_key="output_c", out_key="output_a")
            container.replace_component("A", comp_a_modified)
            container.dependencies["A"] = ["C"]  # Manually create cycle for test
            container._check_circular_dependencies(container.dependencies)


class TestLazyComponentContainer:
    """Test the LazyComponentContainer with deferred initialization."""

    def test_config_registration(self):
        """Test registration of component configurations."""
        container = LazyComponentContainer()

        container.register_component_config(
            name="test_module",
            component_class=LinearModule,
            config={"out_features": 5},
            in_keys=["input"],
            out_keys=["output"],
        )

        assert "test_module" in container.component_configs
        assert container.get_component_dependencies("test_module") == []
        assert container.in_keys == ["input"]
        assert container.out_keys == ["output"]

    def test_initialization_with_shapes(self):
        """Test component initialization with shape inference."""
        container = LazyComponentContainer()

        container.register_component_config(
            name="obs_processor",
            component_class=LinearModule,
            config={"out_features": 8},
            in_keys=["observation"],
            out_keys=["processed_obs"],
        )

        container.register_component_config(
            name="policy",
            component_class=LinearModule,
            config={"out_features": 3},
            dependencies=["obs_processor"],
            in_keys=["processed_obs"],
            out_keys=["action"],
        )

        # Initialize with shapes
        container.initialize_with_input_shapes({"observation": (10,)})

        assert container.initialized
        assert "obs_processor" in container.components
        assert "policy" in container.components

        # Check that components were initialized with correct shapes
        obs_processor = container.components["obs_processor"]
        policy = container.components["policy"]

        # Should be Linear(10, 8) and Linear(8, 3)
        assert obs_processor.linear.in_features == 10
        assert obs_processor.linear.out_features == 8
        assert policy.linear.in_features == 8
        assert policy.linear.out_features == 3

    def test_execution_before_initialization_error(self):
        """Test error when trying to execute before initialization."""
        container = LazyComponentContainer()

        container.register_component_config(
            name="test",
            component_class=LinearModule,
            config={"out_features": 5},
            in_keys=["input"],
            out_keys=["output"],
        )

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        with pytest.raises(RuntimeError, match="must be initialized with input shapes first"):
            container.execute_component("test", td)

        with pytest.raises(RuntimeError, match="must be initialized with input shapes first"):
            container.execute_network(td)

    def test_initialization_error_missing_shape(self):
        """Test error when required input shape is missing."""
        container = LazyComponentContainer()

        container.register_component_config(
            name="test",
            component_class=LinearModule,
            config={"out_features": 5},
            in_keys=["input"],
            out_keys=["output"],
        )

        with pytest.raises(ValueError, match="Shape not specified for container input"):
            container.initialize_with_input_shapes({})  # Missing "input"

    def test_repr_before_and_after_init(self):
        """Test string representation before and after initialization."""
        container = LazyComponentContainer()

        container.register_component_config(
            name="A", component_class=LinearModule, config={"out_features": 5}, in_keys=["input"], out_keys=["output"]
        )

        # Test before initialization
        repr_str = repr(container)
        assert "component configs (not initialized)" in repr_str
        assert "A: LinearModule (deps: [])" in repr_str

        # Test after initialization
        container.initialize_with_input_shapes({"input": (10,)})
        repr_str = repr(container)
        assert "components (initialized)" in repr_str


class TestSafeLazyComponentContainer:
    """Test the combined SafeLazyComponentContainer."""

    def test_invalid_component_class(self):
        """Test validation of component class during config registration."""
        container = SafeLazyComponentContainer()

        # Try to register a class that doesn't inherit from MettaModule
        class BadClass:
            pass

        with pytest.raises(TypeError, match="Component class must inherit from MettaModule"):
            container.register_component_config(
                name="bad", component_class=BadClass, config={}, in_keys=["input"], out_keys=["output"]
            )

    def test_full_pipeline_with_validation(self):
        """Test complete pipeline with both lazy initialization and safety validation."""
        container = SafeLazyComponentContainer()

        # Register configurations
        container.register_component_config(
            name="obs_processor",
            component_class=LinearModule,
            config={"out_features": 8},
            in_keys=["observation"],
            out_keys=["processed_obs"],
        )

        container.register_component_config(
            name="policy",
            component_class=LinearModule,
            config={"out_features": 3},
            dependencies=["obs_processor"],
            in_keys=["processed_obs"],
            out_keys=["action"],
        )

        # Initialize with shapes (includes validation)
        container.initialize_with_input_shapes({"observation": (10,)})

        # Test execution
        td = TensorDict({"observation": torch.randn(2, 10)}, batch_size=2)
        result = container.execute_component("policy", td)

        # Both outputs should be present
        assert "processed_obs" in result
        assert "action" in result
        assert result["processed_obs"].shape == (2, 8)
        assert result["action"].shape == (2, 3)
