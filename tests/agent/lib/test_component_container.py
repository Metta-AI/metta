import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.component_container import ComponentContainer
from metta.agent.lib.metta_moduly import LinearModule


class TestComponentContainer:
    """Test ComponentContainer for agent-level component registry with dependencies."""

    def test_basic_registration(self):
        """Test basic component registration."""
        container = ComponentContainer()
        module = LinearModule(10, 5, "input", "output")

        container.register_component("test_module", module)

        assert "test_module" in container
        assert container["test_module"] is module
        assert container.get_component_dependencies("test_module") == []

    def test_dependency_registration(self):
        """Test component registration with dependencies."""
        container = ComponentContainer()

        obs_processor = LinearModule(10, 8, "observation", "processed_obs")
        policy = LinearModule(8, 3, "processed_obs", "action")

        container.register_component("obs_processor", obs_processor)
        container.register_component("policy", policy, dependencies=["obs_processor"])

        assert container.get_component_dependencies("obs_processor") == []
        assert container.get_component_dependencies("policy") == ["obs_processor"]

    def test_execution_order_calculation(self):
        """Test execution order computation for dependency chains."""
        container = ComponentContainer()

        # Create dependency chain: A -> B -> C
        a = LinearModule(10, 8, "input", "a_out")
        b = LinearModule(8, 6, "a_out", "b_out")
        c = LinearModule(6, 3, "b_out", "output")

        container.register_component("A", a)
        container.register_component("B", b, dependencies=["A"])
        container.register_component("C", c, dependencies=["B"])

        order = container.get_execution_order("C")
        assert order == ["A", "B", "C"]

        order = container.get_execution_order("B")
        assert order == ["A", "B"]

    def test_recursive_execution(self):
        """Test recursive execution of dependencies."""
        container = ComponentContainer()

        obs_processor = LinearModule(10, 8, "observation", "processed_obs")
        policy = LinearModule(8, 3, "processed_obs", "action")

        container.register_component("obs_processor", obs_processor)
        container.register_component("policy", policy, dependencies=["obs_processor"])

        # Test data
        td = TensorDict({"observation": torch.randn(2, 10)}, batch_size=2)

        # Clear cache and execute policy (should recursively execute obs_processor)
        container.clear_cache()
        result = container.forward("policy", td)

        # Both outputs should be present
        assert "processed_obs" in result
        assert "action" in result
        assert result["processed_obs"].shape == (2, 8)
        assert result["action"].shape == (2, 3)

    def test_execution_caching(self):
        """Test that components are not re-executed within same forward pass."""
        container = ComponentContainer()

        # Create a module that tracks execution count
        class CountingModule(LinearModule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.execution_count = 0

            def forward(self, tensordict):
                self.execution_count += 1
                return super().forward(tensordict)

        counting_module = CountingModule(10, 8, "input", "output")
        dependent_module = LinearModule(8, 3, "output", "final")

        container.register_component("counter", counting_module)
        container.register_component("dependent", dependent_module, dependencies=["counter"])

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        # Clear cache and execute dependent twice
        container.clear_cache()
        container.forward("dependent", td)
        container.forward("dependent", td)  # Should use cache

        # Counter should only execute once
        assert counting_module.execution_count == 1

    def test_output_presence_check(self):
        """Test that components skip execution if outputs already exist."""
        container = ComponentContainer()

        module = LinearModule(10, 5, "input", "output")
        container.register_component("test", module)

        # Pre-populate output
        td = TensorDict({"input": torch.randn(2, 10), "output": torch.randn(2, 5)}, batch_size=2)

        container.clear_cache()
        result = container.forward("test", td)

        # Should return original pre-populated output unchanged
        assert torch.equal(result["output"], td["output"])

    def test_hotswapping(self):
        """Test component replacement (hotswapping)."""
        container = ComponentContainer()

        original = LinearModule(10, 5, "input", "output")
        replacement = LinearModule(10, 8, "input", "output")  # Different output size

        container.register_component("swappable", original)

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        # Execute with original
        container.clear_cache()
        result1 = container.forward("swappable", td)
        assert result1["output"].shape == (2, 5)

        # Replace and execute with fresh tensordict
        container.replace_component("swappable", replacement)
        container.clear_cache()

        # Use fresh tensordict to ensure no cached outputs
        td_fresh = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
        result2 = container.forward("swappable", td_fresh)
        assert result2["output"].shape == (2, 8)

    def test_dependency_validation_missing(self):
        """Test validation catches missing dependencies."""
        container = ComponentContainer()

        module = LinearModule(10, 5, "input", "output")
        container.register_component("dependent", module, dependencies=["missing"])

        with pytest.raises(ValueError, match="depends on non-existent component"):
            container.validate_dependencies()

    def test_dependency_validation_cycles(self):
        """Test validation catches circular dependencies."""
        container = ComponentContainer()

        a = LinearModule(10, 8, "input_a", "output_a")
        b = LinearModule(8, 6, "input_b", "output_b")
        c = LinearModule(6, 10, "input_c", "output_c")

        # Create circular dependency: A -> B -> C -> A
        container.register_component("A", a, dependencies=["C"])
        container.register_component("B", b, dependencies=["A"])
        container.register_component("C", c, dependencies=["B"])

        with pytest.raises(ValueError, match="Circular dependency detected"):
            container.validate_dependencies()

    def test_complex_dependency_graph(self):
        """Test complex dependency graph with multiple branches."""
        container = ComponentContainer()

        # Create diamond dependency: A -> {B, C} -> D
        a = LinearModule(10, 8, "input", "a_out")
        b = LinearModule(8, 6, "a_out", "b_out")
        c = LinearModule(8, 6, "a_out", "c_out")
        # D takes concatenated input from B and C (6+6=12)
        d = LinearModule(12, 3, "combined", "output")  # Single input key

        container.register_component("A", a)
        container.register_component("B", b, dependencies=["A"])
        container.register_component("C", c, dependencies=["A"])
        container.register_component("D", d, dependencies=["B", "C"])

        # Validate dependencies
        container.validate_dependencies()

        # Test execution - manually combine B and C outputs for D
        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
        container.clear_cache()

        # Execute B and C first
        container.forward("B", td)
        container.forward("C", td)

        # Manually combine outputs for D input
        combined = torch.cat([td["b_out"], td["c_out"]], dim=1)
        td["combined"] = combined

        result = container.forward("D", td)

        # All outputs should be present
        assert "a_out" in result
        assert "b_out" in result
        assert "c_out" in result
        assert "output" in result

    def test_repr(self):
        """Test string representation."""
        container = ComponentContainer()

        a = LinearModule(10, 5, "input", "output")
        b = LinearModule(5, 3, "output", "final")

        container.register_component("A", a)
        container.register_component("B", b, dependencies=["A"])

        repr_str = repr(container)
        assert "ComponentContainer with 2 components" in repr_str
        assert "A: LinearModule (deps: [])" in repr_str
        assert "B: LinearModule (deps: ['A'])" in repr_str

    def test_cache_clearing(self):
        """Test cache clearing between forward passes."""
        container = ComponentContainer()

        module = LinearModule(10, 5, "input", "output")
        container.register_component("test", module)

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        # Execute once
        container.clear_cache()
        container.forward("test", td)
        assert "test" in container._execution_cache

        # Clear cache
        container.clear_cache()
        assert len(container._execution_cache) == 0

    def test_no_dependencies_execution(self):
        """Test execution of component with no dependencies."""
        container = ComponentContainer()

        module = LinearModule(10, 5, "input", "output")
        container.register_component("standalone", module)

        td = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)

        container.clear_cache()
        result = container.forward("standalone", td)

        assert "output" in result
        assert result["output"].shape == (2, 5)

    def test_execution_order_no_dependencies(self):
        """Test execution order for component with no dependencies."""
        container = ComponentContainer()

        module = LinearModule(10, 5, "input", "output")
        container.register_component("standalone", module)

        order = container.get_execution_order("standalone")
        assert order == ["standalone"]
