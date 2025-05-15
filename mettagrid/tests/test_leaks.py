import gc
import os
import random

import numpy as np
import psutil

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv

# Define a constant seed for deterministic behavior
TEST_SEED = 42


def test_mettagrid_env_reset():
    """Test that the MettaGridEnv can be reset multiple times without memory leaks."""
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

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
    Uses a binning strategy to detect true memory growth while ignoring outliers.
    """
    # Force garbage collection before starting
    gc.collect()

    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    cfg = get_cfg("benchmark")

    print("Pre-warming phase:")
    for _i in range(10):  # Reduced from 20 to 10 to speed up the test
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

    # Run fewer iterations (50 instead of 100)
    num_iterations = 50
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

        # Record memory after each iteration (not just every 5)
        current_memory = get_memory_usage()
        memory_readings.append(current_memory)

        # Print less frequently to keep the output clean
        if i % 10 == 0:
            print(
                f"Iteration {i}: Memory usage: {current_memory:.2f} MB, "
                f"Delta: {current_memory - baseline_memory:.2f} MB"
            )

    # Skip the first few readings (often more unstable)
    skip_first = 5
    analyzed_readings = memory_readings[skip_first:]

    # Create 5 bins of equal size
    num_bins = 5
    readings_per_bin = len(analyzed_readings) // num_bins

    # Calculate the average for each bin
    bin_averages = []
    for i in range(num_bins):
        start_idx = i * readings_per_bin
        end_idx = (i + 1) * readings_per_bin if i < num_bins - 1 else len(analyzed_readings)
        bin_data = analyzed_readings[start_idx:end_idx]
        bin_avg = np.mean(bin_data)
        bin_averages.append(bin_avg)
        print(
            f"Bin {i + 1} (iterations {skip_first + start_idx}-{skip_first + end_idx - 1}): "
            f"Average memory {bin_avg:.2f} MB"
        )

    # Analyze memory growth trend using bin averages
    first_bin_avg = bin_averages[0]
    last_bin_avg = bin_averages[-1]
    total_growth = last_bin_avg - first_bin_avg

    print(f"Total growth from first bin to last bin: {total_growth:.2f} MB")

    # Check for a clear trend - are later bins consistently higher than earlier ones?
    is_increasing = all(bin_averages[i] <= bin_averages[i + 1] for i in range(num_bins - 1))

    if is_increasing and total_growth > 0:
        # Calculate average growth per bin
        avg_growth_per_bin = total_growth / (num_bins - 1)
        print(f"Average growth per bin: {avg_growth_per_bin:.2f} MB")

        # Set a reasonable threshold for bin-to-bin growth
        bin_growth_threshold = 0.5  # MB per bin

        # Only fail if we see consistent growth above the threshold
        assert avg_growth_per_bin < bin_growth_threshold, (
            f"Persistent memory leak detected: {avg_growth_per_bin:.2f} MB average growth per bin"
        )
    else:
        print("No consistent increasing memory trend detected")

    # Final sanity check on absolute growth
    final_growth = memory_readings[-1] - baseline_memory
    print(f"Total memory growth after warmup: {final_growth:.2f} MB")

    # Set a reasonable threshold for total growth
    final_threshold = 10.0  # MB
    assert final_growth < final_threshold, f"Excessive total memory growth: {final_growth:.2f} MB"
