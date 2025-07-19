import pytest
import torch

from metta.rl import mps
from metta.rl.functions.advantage import compute_advantage


class TestAdvantage:
    """Test advantage computation functions"""

    @pytest.fixture
    def advantage_test_data(self):
        """Generate test data for advantage computation"""
        torch.manual_seed(42)  # For reproducibility

        # Test dimensions
        T = 32  # timesteps
        B = 16  # batch size

        # Generate test tensors
        values = torch.randn(T, B)
        rewards = torch.randn(T, B)
        dones = torch.bernoulli(torch.full((T, B), 0.1))  # 10% chance of done
        importance_sampling_ratio = torch.ones(T, B)  # Default to 1.0 for now

        # GAE parameters
        gamma = 0.99
        gae_lambda = 0.95
        vtrace_rho_clip = 1.0
        vtrace_c_clip = 1.0

        return {
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "importance_sampling_ratio": importance_sampling_ratio,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "vtrace_rho_clip": vtrace_rho_clip,
            "vtrace_c_clip": vtrace_c_clip,
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mps_advantage_matches_cuda(self, advantage_test_data):
        """Test that MPS advantage implementation matches CUDA implementation"""
        # Get test data
        data = advantage_test_data

        # Create advantages tensor for CUDA implementation
        advantages_cuda = torch.zeros_like(data["values"])

        # Compute advantage using CUDA implementation
        cuda_device = torch.device("cuda:0")
        advantages_cuda = compute_advantage(
            values=data["values"].clone(),
            rewards=data["rewards"].clone(),
            dones=data["dones"].clone(),
            importance_sampling_ratio=data["importance_sampling_ratio"].clone(),
            advantages=advantages_cuda,
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            device=cuda_device,
        )

        # Compute advantage using MPS implementation directly
        # We'll call the MPS function directly rather than trying to force device='mps'
        advantages_mps = mps.advantage(
            values=data["values"].clone(),
            rewards=data["rewards"].clone(),
            dones=data["dones"].clone(),
            importance_sampling_ratio=data["importance_sampling_ratio"].clone(),
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=torch.device("cpu"),  # MPS implementation can run on CPU for testing
        )

        # Move results to CPU for comparison
        advantages_cuda_cpu = advantages_cuda.cpu()
        advantages_mps_cpu = advantages_mps.cpu()

        # Compare results - should be very close
        assert torch.allclose(advantages_cuda_cpu, advantages_mps_cpu, atol=1e-5, rtol=1e-5), (
            f"MPS and CUDA advantages don't match. Max diff: {(advantages_cuda_cpu - advantages_mps_cpu).abs().max()}"
        )

    def test_mps_advantage_cpu_fallback(self, advantage_test_data):
        """Test that MPS implementation works correctly on CPU"""
        # Get test data
        data = advantage_test_data

        # Test the MPS implementation directly on CPU
        advantages_result = mps.advantage(
            values=data["values"],
            rewards=data["rewards"],
            dones=data["dones"],
            importance_sampling_ratio=data["importance_sampling_ratio"],
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=torch.device("cpu"),
        )

        # Basic sanity checks
        assert advantages_result.shape == data["values"].shape
        assert not torch.isnan(advantages_result).any(), "Advantages contain NaN values"
        assert not torch.isinf(advantages_result).any(), "Advantages contain infinite values"

    def test_mps_advantage_with_vtrace_clipping(self, advantage_test_data):
        """Test MPS advantage with different vtrace clipping values"""
        data = advantage_test_data

        # Test with more aggressive clipping
        data["importance_sampling_ratio"] = torch.rand_like(data["values"]) * 3.0  # Values between 0 and 3
        data["vtrace_rho_clip"] = 1.5
        data["vtrace_c_clip"] = 1.2

        # Compute with MPS implementation directly on CPU
        advantages_result = mps.advantage(
            values=data["values"].clone(),
            rewards=data["rewards"].clone(),
            dones=data["dones"].clone(),
            importance_sampling_ratio=data["importance_sampling_ratio"].clone(),
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=torch.device("cpu"),
        )

        # Verify the result is reasonable
        assert advantages_result.shape == data["values"].shape
        assert not torch.isnan(advantages_result).any()
        assert not torch.isinf(advantages_result).any()

    @pytest.mark.parametrize("T,B", [(16, 8), (64, 32), (128, 4)])
    def test_mps_advantage_different_shapes(self, T, B):
        """Test MPS advantage with different tensor shapes"""
        torch.manual_seed(42)

        # Generate test data with specified shape
        values = torch.randn(T, B)
        rewards = torch.randn(T, B)
        dones = torch.bernoulli(torch.full((T, B), 0.05))
        importance_sampling_ratio = torch.ones(T, B)

        # Compute advantage using MPS implementation directly on CPU
        advantages_result = mps.advantage(
            values=values,
            rewards=rewards,
            dones=dones,
            importance_sampling_ratio=importance_sampling_ratio,
            vtrace_rho_clip=1.0,
            vtrace_c_clip=1.0,
            gamma=0.99,
            gae_lambda=0.95,
            device=torch.device("cpu"),
        )

        # Verify shape and validity
        assert advantages_result.shape == (T, B)
        assert not torch.isnan(advantages_result).any()
        assert not torch.isinf(advantages_result).any()

    def test_mps_advantage_implementation(self, advantage_test_data):
        """Test MPS advantage implementation directly (works on any system)"""
        # Get test data
        data = advantage_test_data

        # Test the MPS implementation directly
        advantages = mps.advantage(
            values=data["values"],
            rewards=data["rewards"],
            dones=data["dones"],
            importance_sampling_ratio=data["importance_sampling_ratio"],
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=torch.device("cpu"),
        )

        # Basic sanity checks
        assert advantages.shape == data["values"].shape, "Advantages shape mismatch"
        assert not torch.isnan(advantages).any(), "Advantages contain NaN values"
        assert not torch.isinf(advantages).any(), "Advantages contain infinite values"

        # Test that advantages are computed correctly for a simple case
        # When all dones are 0 and importance_sampling_ratio is 1,
        # this should follow standard GAE computation
        simple_values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        simple_rewards = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        simple_dones = torch.zeros(4, 1)
        simple_isr = torch.ones(4, 1)

        simple_advantages = mps.advantage(
            values=simple_values,
            rewards=simple_rewards,
            dones=simple_dones,
            importance_sampling_ratio=simple_isr,
            vtrace_rho_clip=1.0,
            vtrace_c_clip=1.0,
            gamma=1.0,  # No discounting for simplicity
            gae_lambda=1.0,  # No lambda discounting
            device=torch.device("cpu"),
        )

        # Manually compute expected advantages for this simple case
        # With gamma=1, lambda=1, no dones, and ISR=1:
        # advantage[t] = sum from i=t to T-2 of (r[i+1] + v[i+1] - v[i])
        expected_adv_2 = 0.0 + 4.0 - 3.0  # t=2: r[3] + v[3] - v[2] = 0 + 4 - 3 = 1
        expected_adv_1 = (1.0 + 3.0 - 2.0) + expected_adv_2  # t=1: delta[1] + adv[2] = 2 + 1 = 3
        expected_adv_0 = (1.0 + 2.0 - 1.0) + expected_adv_1  # t=0: delta[0] + adv[1] = 2 + 3 = 5

        assert torch.allclose(simple_advantages[0], torch.tensor([expected_adv_0]), atol=1e-5), (
            f"Expected {expected_adv_0}, got {simple_advantages[0]}"
        )
        assert torch.allclose(simple_advantages[1], torch.tensor([expected_adv_1]), atol=1e-5), (
            f"Expected {expected_adv_1}, got {simple_advantages[1]}"
        )
        assert torch.allclose(simple_advantages[2], torch.tensor([expected_adv_2]), atol=1e-5), (
            f"Expected {expected_adv_2}, got {simple_advantages[2]}"
        )
        assert torch.allclose(simple_advantages[3], torch.tensor([0.0]), atol=1e-5), (
            f"Expected 0.0, got {simple_advantages[3]}"
        )

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device_support(self, advantage_test_data):
        """Test that the MPS implementation works on actual MPS device when available"""
        data = advantage_test_data

        # Test on MPS device
        mps_device = torch.device("mps")
        advantages_mps = mps.advantage(
            values=data["values"],
            rewards=data["rewards"],
            dones=data["dones"],
            importance_sampling_ratio=data["importance_sampling_ratio"],
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=mps_device,
        )

        # Also compute on CPU for comparison
        advantages_cpu = mps.advantage(
            values=data["values"],
            rewards=data["rewards"],
            dones=data["dones"],
            importance_sampling_ratio=data["importance_sampling_ratio"],
            vtrace_rho_clip=data["vtrace_rho_clip"],
            vtrace_c_clip=data["vtrace_c_clip"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            device=torch.device("cpu"),
        )

        # Results should match
        assert torch.allclose(advantages_mps.cpu(), advantages_cpu, atol=1e-5, rtol=1e-5)
        assert advantages_mps.device.type == "mps"
