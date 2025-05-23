import gc
import os
import time

import psutil
import pytest
import torch
from tensordict import TensorDict

from metta.agent.lib.component_container import ComponentContainer
from metta.agent.lib.metta_modules import (
    Conv2dModule,
    LinearModule,
    MettaModule,
    ReLUModule,
)
from metta.agent.lib.wrapper_modules import RegularizedModule, SafeModule, WeightMonitoringModule


class TestPerformance:
    """Performance and stress tests to ensure modules scale well."""

    def test_linear_module_performance(self):
        """Test LinearModule performance with various sizes."""
        sizes = [(100, 50), (1000, 500), (5000, 1000)]
        batch_sizes = [32, 128, 512]

        for in_features, out_features in sizes:
            module = LinearModule(in_features=in_features, out_features=out_features, in_key="input", out_key="output")
            module.eval()  # Consistent timing

            for batch_size in batch_sizes:
                td = TensorDict({"input": torch.randn(batch_size, in_features)}, batch_size=batch_size)

                # Warm up
                for _ in range(5):
                    _ = module(td.clone())

                # Time execution
                start_time = time.time()
                iterations = 100

                for _ in range(iterations):
                    result = module(td.clone())

                end_time = time.time()
                avg_time = (end_time - start_time) / iterations

                # Performance should be reasonable (< 10ms for largest case)
                assert avg_time < 0.01, (
                    f"Too slow: {avg_time:.4f}s for {in_features}x{out_features}, batch {batch_size}"
                )

    def test_conv_module_performance(self):
        """Test Conv2dModule performance with realistic image sizes."""
        configurations = [
            (3, 32, 3, 32, 32),  # Small image
            (32, 64, 3, 64, 64),  # Medium image
            (64, 128, 3, 128, 128),  # Large image
        ]

        for in_ch, out_ch, kernel, height, width in configurations:
            module = Conv2dModule(in_ch, out_ch, kernel, padding=1)
            module.eval()

            batch_size = 16
            td = TensorDict({"input": torch.randn(batch_size, in_ch, height, width)}, batch_size=batch_size)

            # Warm up
            for _ in range(3):
                _ = module(td.clone())

            # Time execution
            start_time = time.time()
            iterations = 50

            for _ in range(iterations):
                result = module(td.clone())

            end_time = time.time()
            avg_time = (end_time - start_time) / iterations

            # Should complete in reasonable time
            assert avg_time < 0.1, f"Conv too slow: {avg_time:.4f}s for {in_ch}→{out_ch}, {height}x{width}"

    def test_component_container_performance(self):
        """Test ComponentContainer performance with complex dependency graphs."""
        container = ComponentContainer()

        # Build a complex multi-branch network
        # input -> [branch1, branch2, branch3] -> combiner -> output

        branch1 = [
            LinearModule(in_features=100, out_features=64, in_key="input", out_key="b1_h1"),
            ReLUModule("b1_h1", "b1_h2"),
            LinearModule(in_features=64, out_features=32, in_key="b1_h2", out_key="branch1_out"),
        ]

        branch2 = [
            LinearModule(in_features=100, out_features=128, in_key="input", out_key="b2_h1"),
            ReLUModule("b2_h1", "b2_h2"),
            LinearModule(in_features=128, out_features=32, in_key="b2_h2", out_key="branch2_out"),
        ]

        branch3 = [
            LinearModule(in_features=100, out_features=96, in_key="input", out_key="b3_h1"),
            ReLUModule("b3_h1", "b3_h2"),
            LinearModule(in_features=96, out_features=32, in_key="b3_h2", out_key="branch3_out"),
        ]

        # Combiner takes all branch outputs
        class CombinerModule(MettaModule):
            def __init__(self):
                super().__init__(in_keys=["branch1_out", "branch2_out", "branch3_out"], out_keys=["final_output"])
                self.combiner = torch.nn.Linear(96, 10)  # 3 * 32 = 96

            def forward(self, tensordict):
                combined = torch.cat(
                    [tensordict["branch1_out"], tensordict["branch2_out"], tensordict["branch3_out"]], dim=1
                )
                tensordict["final_output"] = self.combiner(combined)
                return tensordict

        combiner = CombinerModule()

        # Register all components
        # Branch 1
        container.register_component("b1_linear1", branch1[0])
        container.register_component("b1_relu", branch1[1], dependencies=["b1_linear1"])
        container.register_component("b1_linear2", branch1[2], dependencies=["b1_relu"])

        # Branch 2
        container.register_component("b2_linear1", branch2[0])
        container.register_component("b2_relu", branch2[1], dependencies=["b2_linear1"])
        container.register_component("b2_linear2", branch2[2], dependencies=["b2_relu"])

        # Branch 3
        container.register_component("b3_linear1", branch3[0])
        container.register_component("b3_relu", branch3[1], dependencies=["b3_linear1"])
        container.register_component("b3_linear2", branch3[2], dependencies=["b3_relu"])

        # Combiner depends on all branches
        container.register_component("combiner", combiner, dependencies=["b1_linear2", "b2_linear2", "b3_linear2"])

        # Test performance
        td = TensorDict({"input": torch.randn(32, 100)}, batch_size=32)

        # Warm up
        for _ in range(3):
            container.clear_cache()
            _ = container.execute_component("combiner", td.clone())

        # Time execution
        start_time = time.time()
        iterations = 50

        for _ in range(iterations):
            container.clear_cache()
            result = container.execute_component("combiner", td.clone())

        end_time = time.time()
        avg_time = (end_time - start_time) / iterations

        # Complex graph should still be fast
        assert avg_time < 0.05, f"Complex container too slow: {avg_time:.4f}s"
        assert result["final_output"].shape == (32, 10)

    def test_wrapper_module_overhead(self):
        """Test performance overhead of wrapper modules."""
        base_module = LinearModule(in_features=1000, out_features=500, in_key="input", out_key="output")

        # Test different wrapper combinations
        wrappers = {
            "base": base_module,
            "safe": SafeModule(base_module),
            "regularized": RegularizedModule(base_module, l2_scale=0.01),
            "monitored": WeightMonitoringModule(base_module, monitor_health=True),
            "triple_wrapped": WeightMonitoringModule(
                RegularizedModule(SafeModule(base_module), l2_scale=0.01), monitor_health=True
            ),
        }

        td = TensorDict({"input": torch.randn(64, 1000)}, batch_size=64)
        results = {}

        for name, module in wrappers.items():
            module.eval()

            # Warm up
            for _ in range(5):
                _ = module(td.clone())

            # Time execution
            start_time = time.time()
            iterations = 100

            for _ in range(iterations):
                result = module(td.clone())

            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            results[name] = avg_time

        # Wrapper overhead should be minimal (< 50% increase)
        base_time = results["base"]
        for name, time_taken in results.items():
            if name != "base":
                overhead = (time_taken - base_time) / base_time
                assert overhead < 0.5, f"{name} wrapper adds too much overhead: {overhead:.2%}"

    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        if not hasattr(psutil, "Process"):
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and use many modules
        modules = []
        for i in range(100):
            module = LinearModule(in_features=100, out_features=50, in_key="input", out_key="output")
            modules.append(module)

            # Use the module
            td = TensorDict({"input": torch.randn(10, 100)}, batch_size=10)
            _ = module(td)

        # Check memory usage
        gc.collect()
        used_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = used_memory - baseline_memory

        # Memory usage should be reasonable (< 500MB for 100 small modules)
        assert memory_increase < 500, f"Too much memory used: {memory_increase:.1f}MB"

        # Clean up
        del modules
        gc.collect()

    def test_batch_size_scaling(self):
        """Test how performance scales with batch size."""
        module = LinearModule(in_features=512, out_features=256, in_key="input", out_key="output")
        module.eval()

        batch_sizes = [1, 16, 64, 256, 1024]
        times = []

        for batch_size in batch_sizes:
            td = TensorDict({"input": torch.randn(batch_size, 512)}, batch_size=batch_size)

            # Warm up
            for _ in range(3):
                _ = module(td.clone())

            # Time execution
            start_time = time.time()
            iterations = 20

            for _ in range(iterations):
                result = module(td.clone())

            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            times.append(avg_time)

        # Performance should scale reasonably (not more than linear in batch size)
        # Larger batches should be more efficient per sample
        time_per_sample = [t / bs for t, bs in zip(times, batch_sizes, strict=False)]

        # Later batch sizes should be more efficient (lower time per sample)
        assert time_per_sample[-1] < time_per_sample[0], "Batch processing should be more efficient"

    def test_deep_network_performance(self):
        """Test performance with very deep networks."""
        depth = 50
        feature_size = 128

        # Create deep network
        modules = []
        for i in range(depth):
            in_key = "input" if i == 0 else f"layer_{i - 1}"
            out_key = "output" if i == depth - 1 else f"layer_{i}"

            # Alternate between linear and ReLU
            if i % 2 == 0:
                modules.append(
                    LinearModule(in_features=feature_size, out_features=feature_size, in_key=in_key, out_key=out_key)
                )
            else:
                modules.append(ReLUModule(in_key, out_key))

        td = TensorDict({"input": torch.randn(32, feature_size)}, batch_size=32)

        # Warm up
        for _ in range(3):
            result = td.clone()
            for module in modules:
                result = module(result)

        # Time execution
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            result = td.clone()
            for module in modules:
                result = module(result)

        end_time = time.time()
        avg_time = (end_time - start_time) / iterations

        # Deep network should complete in reasonable time
        assert avg_time < 0.1, f"Deep network too slow: {avg_time:.4f}s for {depth} layers"
        assert result["output"].shape == (32, feature_size)

    def test_concurrent_execution(self):
        """Test performance with concurrent module execution."""
        import queue
        import threading

        module = LinearModule(in_features=256, out_features=128, in_key="input", out_key="output")
        module.eval()

        # Shared queue for results
        result_queue = queue.Queue()

        def worker():
            """Worker function for concurrent execution."""
            td = TensorDict({"input": torch.randn(16, 256)}, batch_size=16)

            start_time = time.time()
            for _ in range(10):
                result = module(td.clone())
            end_time = time.time()

            result_queue.put(end_time - start_time)

        # Run multiple workers concurrently
        num_workers = 4
        threads = []

        start_time = time.time()
        for _ in range(num_workers):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        worker_times = []
        while not result_queue.empty():
            worker_times.append(result_queue.get())

        # Concurrent execution should be reasonably efficient
        avg_worker_time = sum(worker_times) / len(worker_times)
        assert total_time < avg_worker_time * 2, "Concurrent execution has too much overhead"

    def test_gradient_computation_performance(self):
        """Test performance of gradient computation."""
        modules = [
            LinearModule(in_features=512, out_features=256, in_key="input", out_key="hidden1"),
            ReLUModule("hidden1", "relu1"),
            LinearModule(in_features=256, out_features=128, in_key="relu1", out_key="hidden2"),
            ReLUModule("hidden2", "relu2"),
            LinearModule(in_features=128, out_features=1, in_key="relu2", out_key="output"),
        ]

        # Set requires_grad for all parameters
        for module in modules:
            for param in module.parameters():
                param.requires_grad_(True)

        td = TensorDict({"input": torch.randn(64, 512, requires_grad=True)}, batch_size=64)

        # Warm up
        for _ in range(3):
            test_td = td.clone()
            result = test_td
            for module in modules:
                result = module(result)
            loss = result["output"].sum()
            loss.backward()

            # Clear gradients
            for module in modules:
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            test_td["input"].grad = None

        # Time gradient computation
        start_time = time.time()
        iterations = 20

        for _ in range(iterations):
            test_td = td.clone()
            result = test_td
            for module in modules:
                result = module(result)
            loss = result["output"].sum()
            loss.backward()

            # Clear gradients for next iteration
            for module in modules:
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            test_td["input"].grad = None

        end_time = time.time()
        avg_time = (end_time - start_time) / iterations

        # Gradient computation should be reasonably fast
        assert avg_time < 0.05, f"Gradient computation too slow: {avg_time:.4f}s"

    def test_cache_efficiency(self):
        """Test ComponentContainer cache efficiency."""
        container = ComponentContainer()

        # Create modules with caching opportunities
        expensive_module = LinearModule(in_features=1000, out_features=1000, in_key="input", out_key="expensive_output")
        cheap_module = LinearModule(in_features=10, out_features=5, in_key="expensive_output", out_key="final_output")

        container.register_component("expensive", expensive_module)
        container.register_component("cheap", cheap_module, dependencies=["expensive"])

        td = TensorDict({"input": torch.randn(32, 1000)}, batch_size=32)

        # First execution (no cache)
        start_time = time.time()
        result1 = container.execute_component("cheap", td.clone())
        first_time = time.time() - start_time

        # Second execution (should use cache for expensive computation)
        start_time = time.time()
        result2 = container.execute_component("cheap", td.clone())
        second_time = time.time() - start_time

        # Cached execution should be faster
        assert second_time < first_time * 0.8, "Cache not providing expected speedup"

        # Results should be identical
        assert torch.allclose(result1["final_output"], result2["final_output"])

    @pytest.mark.slow
    def test_stress_large_batch(self):
        """Stress test with very large batch sizes."""
        module = LinearModule(in_features=100, out_features=50, in_key="input", out_key="output")
        module.eval()

        # Very large batch
        large_batch_size = 10000
        td = TensorDict({"input": torch.randn(large_batch_size, 100)}, batch_size=large_batch_size)

        # Should handle large batches without error
        start_time = time.time()
        result = module(td)
        execution_time = time.time() - start_time

        assert result["output"].shape == (large_batch_size, 50)
        # Should complete in reasonable time even for large batch
        assert execution_time < 5.0, f"Large batch too slow: {execution_time:.2f}s"

    @pytest.mark.slow
    def test_stress_many_components(self):
        """Stress test with many components in container."""
        container = ComponentContainer()

        # Create many components
        num_components = 200
        for i in range(num_components):
            module = LinearModule(in_features=10, out_features=10, in_key="input", out_key=f"output_{i}")
            container.register_component(f"component_{i}", module)

        td = TensorDict({"input": torch.randn(8, 10)}, batch_size=8)

        # Execute random components
        import random

        for _ in range(20):
            component_name = f"component_{random.randint(0, num_components - 1)}"
            container.clear_cache()

            start_time = time.time()
            result = container.execute_component(component_name, td.clone())
            execution_time = time.time() - start_time

            # Each execution should be fast
            assert execution_time < 0.01, f"Component execution too slow: {execution_time:.4f}s"
            assert result[f"output_{component_name.split('_')[1]}"].shape == (8, 10)
