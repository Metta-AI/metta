import gc
import os

import psutil
import pytest
from hydra import compose, initialize

from metta.mettagrid.curriculum import single_task
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
    curriculum = single_task("test", cfg)
    env = MettaGridEnv(curriculum, render_mode=None)
    assert env is not None, "Failed to initialize MettaGridEnv"


def test_mettagrid_env_reset(cfg):
    """Test that the MettaGridEnv can be reset multiple times without memory leaks."""
    curriculum = single_task("test", cfg)
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
    num_iterations = 20  # Increase this for more thorough testing
    memory_usage = []

    for i in range(num_iterations):
        # Create the environment
        curriculum = single_task("test", cfg)
        env = MettaGridEnv(curriculum, render_mode=None)

        # Reset the environment multiple times
        for _ in range(5):
            observation = env.reset()
            assert observation is not None, "Reset should return a valid observation"

        # Explicitly delete the environment to release resources
        del env

        # Force garbage collection
        gc.collect()

        # Record memory usage
        current_memory = get_memory_usage()
        memory_usage.append(current_memory)
        print(f"Iteration {i + 1}: Memory usage: {current_memory:.2f} MB")

    # Calculate memory growth
    memory_growth = memory_usage[-1] - initial_memory

    # Final memory should not be significantly higher than initial memory
    # Allow for some small fluctuations
    memory_threshold = 5.0  # MB - adjust based on your environment's expected behavior

    print(f"Memory growth after {num_iterations} iterations: {memory_growth:.2f} MB")
    assert memory_growth < memory_threshold, f"Possible memory leak detected: {memory_growth:.2f} MB growth"
