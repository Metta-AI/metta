import gc
import os

import psutil
import pytest
from hydra import compose, initialize

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv


@pytest.fixture(scope="module")
def cfg():
    # Initialize Hydra with the correct relative path
    with initialize(version_base=None, config_path="../configs"):
        # Load the default config
        cfg = compose(config_name="test_basic")
        yield cfg


def test_mettagrid_env_init(cfg):
    """Test that the MettaGridEnv can be initialized properly."""
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)
    assert env is not None, "Failed to initialize MettaGridEnv"


def test_mettagrid_env_reset(cfg):
    """Test that the MettaGridEnv can be reset multiple times without memory leaks."""
    curriculum = SingleTaskCurriculum("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)
    # Reset the environment multiple times
    for _ in range(10):
        observation = env.reset()
        assert observation is not None, "Reset should return a valid observation"


def get_memory_usage():
    """Get the current memory usage of the Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def test_mettagrid_env_no_memory_leaks(cfg):
    """
    Test that the MettaGridEnv can be reset multiple times without memory leaks.

    This test creates and destroys an environment object after multiple resets
    to verify that no memory leaks occur during this process.
    """
    # Force garbage collection before starting
    gc.collect()

    # Get initial memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Create multiple environments and reset them many times
    num_iterations = 50  # Increased for more thorough testing
    memory_usage = []
    max_memory = initial_memory

    for i in range(num_iterations):
        # Create the environment
        curriculum = SingleTaskCurriculum("test", cfg)
        env = MettaGridEnv(curriculum, render_mode=None)

        # Reset the environment multiple times
        for _ in range(5):
            observation = env.reset()
            assert observation is not None, "Reset should return a valid observation"

        # Also run a few steps to test step memory usage
        for _ in range(10):
            actions = env.action_space.sample()
            env.step(actions)

        # Explicitly delete the environment to release resources
        del env
        del curriculum

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Record memory usage
        current_memory = get_memory_usage()
        memory_usage.append(current_memory)
        max_memory = max(max_memory, current_memory)
        
        # Print detailed info every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: Memory usage: {current_memory:.2f} MB (Peak: {max_memory:.2f} MB)")

    # Calculate memory growth and peak usage
    memory_growth = memory_usage[-1] - initial_memory
    peak_growth = max_memory - initial_memory

    # Final memory should not be significantly higher than initial memory
    memory_threshold = 10.0  # MB - increased threshold for more rigorous testing
    peak_threshold = 20.0  # MB - allow for temporary spikes

    print(f"Memory growth after {num_iterations} iterations: {memory_growth:.2f} MB")
    print(f"Peak memory growth: {peak_growth:.2f} MB")
    
    assert memory_growth < memory_threshold, f"Possible memory leak detected: {memory_growth:.2f} MB growth"
    assert peak_growth < peak_threshold, f"Excessive peak memory usage: {peak_growth:.2f} MB growth"


def test_mettagrid_env_stress_test(cfg):
    """
    Stress test with rapid environment creation/destruction to catch edge case leaks.
    """
    # Force garbage collection before starting
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Stress test initial memory: {initial_memory:.2f} MB")

    # Rapid creation/destruction cycles
    num_cycles = 100
    for i in range(num_cycles):
        curriculum = SingleTaskCurriculum("test", cfg)
        env = MettaGridEnv(curriculum, render_mode=None)
        
        # Single reset and step
        env.reset()
        actions = env.action_space.sample()
        env.step(actions)
        
        # Immediate cleanup
        del env
        del curriculum
        
        # Periodic GC and memory check
        if i % 25 == 0:
            gc.collect()
            current_memory = get_memory_usage()
            growth = current_memory - initial_memory
            print(f"Stress cycle {i}: {current_memory:.2f} MB (+{growth:.2f} MB)")
            
            # Early failure detection
            if growth > 50.0:  # 50MB growth is concerning
                pytest.fail(f"Excessive memory growth during stress test: {growth:.2f} MB at cycle {i}")

    # Final check
    gc.collect()
    final_memory = get_memory_usage()
    stress_growth = final_memory - initial_memory
    
    print(f"Stress test final memory growth: {stress_growth:.2f} MB")
    assert stress_growth < 15.0, f"Stress test memory leak: {stress_growth:.2f} MB growth"
