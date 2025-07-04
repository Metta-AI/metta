"""
Unit tests and benchmarks for the compute_gae C++ function.

This module provides comprehensive testing and benchmarking for the Generalized
Advantage Estimation (GAE) implementation in C++, comparing it against a pure
NumPy implementation for correctness and performance.
"""

import numpy as np
import pytest

from metta.rl.fast_gae import compute_gae


def numpy_compute_gae(dones, values, rewards, gamma, gae_lambda):
    """Pure NumPy implementation of GAE for testing purposes.

    Args:
        dones: Binary flags indicating episode termination (1.0 for done, 0.0 for not done)
        values: Value function estimates at each timestep
        rewards: Rewards at each timestep
        gamma: Discount factor
        gae_lambda: GAE lambda parameter for advantage estimation

    Returns:
        advantages: Calculated advantage values
    """
    length = len(dones)
    advantages = np.zeros_like(values)
    last_gae_lam = 0.0

    for t in reversed(range(length - 1)):
        next_non_terminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    return advantages


# Fixtures for test parameters
@pytest.fixture(scope="module")
def gae_params():
    """Provides standard GAE hyperparameters."""
    return {"gamma": 0.99, "gae_lambda": 0.95}


# Fixtures for test data
@pytest.fixture(scope="function")
def simple_trajectory():
    """Basic trajectory for testing GAE calculation."""
    return {
        "dones": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "rewards": np.array([0.5, 0.5, 0.5, 10.0], dtype=np.float32),
        "expected_with_gamma_0_99": np.array([-2.005, -2.5, 0.0, 0.0], dtype=np.float32),
    }


@pytest.fixture(scope="function")
def multi_episode_trajectory():
    """Trajectory with multiple episodes to test episode boundary handling."""
    return {
        "dones": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "values": np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32),
        "rewards": np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32),
    }


@pytest.fixture(scope="function")
def zeros_trajectory():
    """Edge case with all zeros to test numerical stability."""
    return {
        "dones": np.zeros(5, dtype=np.float32),
        "values": np.zeros(5, dtype=np.float32),
        "rewards": np.zeros(5, dtype=np.float32),
    }


@pytest.fixture(scope="function")
def constant_trajectory():
    """Trajectory with constant values for testing different gamma/lambda effects."""
    return {
        "dones": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "values": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "rewards": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }


@pytest.fixture(scope="function")
def mismatched_arrays():
    """Arrays with mismatched lengths for testing input validation."""
    return {
        "dones": np.array([0.0, 0.0], dtype=np.float32),
        "values": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "rewards": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    }


# Benchmark fixtures with different sizes
@pytest.fixture(scope="module")
def make_trajectory():
    """Factory fixture to create trajectories of variable size."""

    def _make_trajectory(size, gamma=0.99, gae_lambda=0.95, random_values=False):
        dones = np.zeros(size, dtype=np.float32)
        dones[-1] = 1.0

        if random_values:
            # Use fixed seed for reproducibility
            rng = np.random.RandomState(42)
            values = rng.normal(0, 1, size=size).astype(np.float32)
            rewards = rng.normal(0, 1, size=size).astype(np.float32)
        else:
            values = np.ones(size, dtype=np.float32)
            rewards = np.ones(size, dtype=np.float32)

        return {"dones": dones, "values": values, "rewards": rewards, "gamma": gamma, "gae_lambda": gae_lambda}

    return _make_trajectory


@pytest.fixture(scope="function")
def small_trajectory(make_trajectory):
    """Small trajectory (100 timesteps) for benchmarking."""
    return make_trajectory(100)


@pytest.fixture(scope="function")
def medium_trajectory(make_trajectory):
    """Medium trajectory (1000 timesteps) for benchmarking."""
    return make_trajectory(1000)


@pytest.fixture(scope="function")
def large_trajectory(make_trajectory):
    """Large trajectory (10000 timesteps) for benchmarking."""
    return make_trajectory(10000)


@pytest.fixture(scope="function")
def realistic_rl_batch(gae_params):
    """Realistic batch size in RL (e.g., PPO with 8 parallel environments, 128 steps)."""
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(42)
    batch_size = 1024  # 8 envs * 128 steps
    dones = np.zeros(batch_size, dtype=np.float32)
    # Add episode terminations at the end of each environment trajectory
    dones[127::128] = 1.0

    return {
        "dones": dones,
        "values": rng.normal(0, 1, size=batch_size).astype(np.float32),
        "rewards": rng.normal(0, 1, size=batch_size).astype(np.float32),
        "gamma": gae_params["gamma"],
        "gae_lambda": gae_params["gae_lambda"],
    }


# Basic test cases
class TestComputeGAE:
    """Test suite for the compute_gae function."""

    def test_basic_functionality(self, simple_trajectory, gae_params):
        """Test basic GAE calculation on a simple trajectory."""
        data = simple_trajectory
        gamma = gae_params["gamma"]
        gae_lambda = gae_params["gae_lambda"]

        # Compute with both implementations
        expected = numpy_compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)
        advantages = compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)

        # Check results match
        np.testing.assert_allclose(advantages, expected, rtol=1e-5)

    def test_episode_boundaries(self, multi_episode_trajectory, gae_params):
        """Test that episode boundaries are handled correctly."""
        data = multi_episode_trajectory
        gamma = gae_params["gamma"]
        gae_lambda = gae_params["gae_lambda"]

        expected = numpy_compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)
        advantages = compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)

        np.testing.assert_allclose(advantages, expected, rtol=1e-5)

        # Verify that advantages don't propagate across episode boundaries
        # When done[t]=1, the advantage for step t shouldn't affect step t-1
        # For our test case with dones=[0,1,0,1], we specifically check that
        # the advantage at index 0 is calculated independently from index 2
        # First calculate the direct advantage without lambda
        next_non_terminal = 1.0 - data["dones"][1]  # 0.0 since done[1]=1.0
        # delta_0 = r_0 + gamma*V(s_1)*next_non_terminal - V(s_0)
        # = 0.5 + 0.99*2.0*0.0 - 1.0 = -0.5
        delta_0 = data["rewards"][0] + gamma * data["values"][1] * next_non_terminal - data["values"][0]
        assert abs(advantages[0] - delta_0) < 1e-5, "Advantage propagated across episode boundary"

    def test_zeros_edge_case(self, zeros_trajectory, gae_params):
        """Test edge case with all zeros to check numerical stability."""
        data = zeros_trajectory
        gamma = gae_params["gamma"]
        gae_lambda = gae_params["gae_lambda"]

        expected = numpy_compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)
        advantages = compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)

        np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        np.testing.assert_allclose(advantages, np.zeros_like(advantages), rtol=1e-5)

    def test_gamma_lambda_effects(self, constant_trajectory):
        """Test effects of different gamma and lambda values."""
        data = constant_trajectory

        # Case 1: With gamma=1.0, lambda=1.0 (advantage should sum future rewards minus value)
        expected1 = numpy_compute_gae(data["dones"], data["values"], data["rewards"], 1.0, 1.0)
        advantages1 = compute_gae(data["dones"], data["values"], data["rewards"], 1.0, 1.0)

        np.testing.assert_allclose(advantages1, expected1, rtol=1e-5)
        # Theoretically, with gamma=1, lambda=1, constant rewards=1, values=1:
        # A[0] = (r0 - v0) + (r1 - v1) + (r2 - v2) + (r3 - v3) = (1-1)+(1-1)+(1-1)+(1-1) = 0
        # This is a special case where advantage is zero when reward=value everywhere
        # But in most cases when values are not equal to rewards, we'd get non-zero advantages

        # Let's modify values to test this better
        modified_values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        expected_advantages = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

        actual_advantages = compute_gae(data["dones"], modified_values, data["rewards"], 1.0, 1.0)
        np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5)

        # Case 2: With gamma=0.0 (only immediate rewards matter)
        expected2 = numpy_compute_gae(data["dones"], data["values"], data["rewards"], 0.0, 0.95)
        advantages2 = compute_gae(data["dones"], data["values"], data["rewards"], 0.0, 0.95)

        np.testing.assert_allclose(advantages2, expected2, rtol=1e-5)
        # With gamma=0, we expect: advantage_t = reward_t - value_t
        expected_gamma0 = data["rewards"] - data["values"]
        np.testing.assert_allclose(advantages2, expected_gamma0, rtol=1e-5)

    def test_input_validation(self, mismatched_arrays, gae_params):
        """Test that input validation works correctly."""
        data = mismatched_arrays
        gamma = gae_params["gamma"]
        gae_lambda = gae_params["gae_lambda"]

        with pytest.raises(RuntimeError, match="same length"):
            compute_gae(data["dones"], data["values"], data["rewards"], gamma, gae_lambda)

    def test_long_trajectory(self, make_trajectory):
        """Test that a long trajectory behaves as expected."""
        # Create a trajectory with 5000 timesteps
        data = make_trajectory(5000)

        # Compute with both implementations
        expected = numpy_compute_gae(data["dones"], data["values"], data["rewards"], data["gamma"], data["gae_lambda"])
        advantages = compute_gae(data["dones"], data["values"], data["rewards"], data["gamma"], data["gae_lambda"])

        np.testing.assert_allclose(advantages, expected, rtol=1e-5)

        # The last advantage should be zero (terminal state)
        assert advantages[-1] == 0.0

        # Test that advantage values decrease as we go farther into the future
        # (with constant rewards=1, values=1, and gamma<1, advantages decrease)
        for i in range(len(advantages) - 2):
            assert advantages[i] > advantages[i + 1] or np.isclose(advantages[i], advantages[i + 1])
