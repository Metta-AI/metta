import gc
import os
import random

import numpy as np
import psutil

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.resolvers import register_resolvers

# Define a constant seed for deterministic behavior
TEST_SEED = 42


def test_mettagrid_env_reset():
    """Test that the MettaGridEnv can be reset multiple times without memory leaks."""
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    register_resolvers()
    cfg = get_cfg("benchmark")

    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    # Reset the environment multiple times
    for _ in range(10):
        obs, infos = env.reset()
        assert obs is not None, "Reset should return a valid observation array"
        assert infos is not None, "Reset should return a valid info dict"


def get_memory_usage():
    """Get the current memory usage of the Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def test_mettagrid_env_no_memory_leaks():
    """
    Test that the MettaGridEnv can be reset multiple times without memory leaks.
    """
    # Force garbage collection before starting
    gc.collect()

    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    register_resolvers()
    cfg = get_cfg("benchmark")

    print("Pre-warming phase:")
    for _i in range(20):
        env = MettaGridEnv(env_cfg=cfg, render_mode=None)
        _obs, _infos = env.reset()
        if hasattr(env, "close"):
            env.close()
        del env
        gc.collect()

    post_warmup_memory = get_memory_usage()
    print(f"Memory after pre-warming: {post_warmup_memory:.2f} MB")
    print(f"Initial warmup cost: {post_warmup_memory - initial_memory:.2f} MB")

    # Use this as our new baseline
    baseline_memory = post_warmup_memory

    # Run a long test to detect stabilization
    num_iterations = 100
    memory_readings = []

    for i in range(num_iterations):
        env = MettaGridEnv(env_cfg=cfg, render_mode=None)

        # Do multiple resets
        for _j in range(3):
            _obs, _infos = env.reset()

        if hasattr(env, "close"):
            env.close()
        del env
        gc.collect()

        # Record memory every 5 iterations
        if i % 5 == 0:
            current_memory = get_memory_usage()
            memory_readings.append(current_memory)
            print(
                f"Iteration {i}: Memory usage: {current_memory:.2f} MB, "
                f"Delta: {current_memory - baseline_memory:.2f} MB"
            )

    # Analyze the last half of the readings to detect stabilization
    if len(memory_readings) >= 10:
        second_half = memory_readings[len(memory_readings) // 2 :]

        # Calculate the slope of the second half
        x = np.array(range(len(memory_readings) // 2 * 5, num_iterations, 5))
        y = np.array(second_half)

        if len(x) >= 2:  # Need at least 2 points for slope calculation
            # Calculate slope (MB per iteration)
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x)) ** 2)

            print(f"Memory growth slope (second half): {slope:.6f} MB per iteration")

            # Check if memory usage has stabilized in the second half
            max_diff = max(second_half) - min(second_half)
            print(f"Max memory fluctuation in second half: {max_diff:.2f} MB")

            # A real leak would show consistent growth even in the second half
            stabilization_threshold = 0.01  # MB per iteration for second half
            assert abs(slope) < stabilization_threshold, (
                f"Persistent memory leak detected: {slope:.6f} MB growth per iteration in second half"
            )

            fluctuation_threshold = 2.0  # MB
            assert max_diff < fluctuation_threshold, f"Excessive memory fluctuation in second half: {max_diff:.2f} MB"

    # Overall growth sanity check
    final_growth = memory_readings[-1] - baseline_memory
    print(f"Total memory growth after warmup: {final_growth:.2f} MB")

    # Set realistic threshold based on observed ~4MB growth and plateauing
    final_threshold = 8.0  # MB
    assert final_growth < final_threshold, f"Excessive total memory growth: {final_growth:.2f} MB"
