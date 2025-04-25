import pytest
import torch

from metta.agent.util.distribution_utils import entropy, log_prob, normalize_logits, sample_logits


@pytest.fixture
def sample_logits_data():
    """Create sample logits of various shapes for testing."""
    batch_size = 3
    vocab_size = 5

    # Single batch, single token
    single_logits = torch.tensor([[1.0, 2.0, 0.5, -1.0, 0.0]])

    # Multiple batches, single token
    batch_logits = torch.randn(batch_size, vocab_size)

    # Create an almost deterministic distribution for testing
    deterministic_logits = torch.tensor([[-1000.0, 1000.0, -1000.0, -1000.0, -1000.0]])

    return {"single": single_logits, "batch": batch_logits, "deterministic": deterministic_logits}


class TestLogProb:
    """Test the log_prob function."""

    def test_single_value(self):
        """Test log_prob with a single value."""
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        value = torch.tensor([1])  # Index 1 corresponds to value 2.0

        # Calculate normalized logits and expected log probability manually
        normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        expected = normalized_logits[0, 1]

        result = log_prob(normalized_logits, value)

        assert torch.isclose(result, expected)

    def test_batch_values(self):
        """Test log_prob with batch values."""
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]])
        values = torch.tensor([0, 2])  # Index 0 for first batch, 2 for second

        # Calculate normalized logits and expected log probabilities manually
        normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        expected = torch.tensor([normalized_logits[0, 0], normalized_logits[1, 2]])

        result = log_prob(normalized_logits, values)

        assert torch.allclose(result, expected)

    def test_broadcasting(self):
        """Test that broadcasting works correctly in log_prob."""
        logits = torch.tensor([[1.0, 2.0, 0.5]])  # Shape: [1, 3]
        values = torch.tensor([1, 2])  # Shape: [2]

        # The broadcast should result in logits being replicated for each value
        normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        expected = torch.tensor([normalized_logits[0, 1], normalized_logits[0, 2]])

        result = log_prob(normalized_logits, values)

        assert torch.allclose(result, expected)

    def test_edge_cases(self):
        """Test edge cases for log_prob function."""
        # Test with very large logits (approaching deterministic)
        large_logits = torch.tensor([[1000.0, -1000.0, -1000.0]])
        large_normalized = large_logits - large_logits.logsumexp(dim=-1, keepdim=True)
        value = torch.tensor([0])

        result = log_prob(large_normalized, value)
        # Should be close to 0 in log space (probability close to 1)
        assert result.item() <= 0 and result.item() > -1e-5

        # Test with very small logits
        small_logits = torch.tensor([[-1000.0, 1000.0, -1000.0]])
        small_normalized = small_logits - small_logits.logsumexp(dim=-1, keepdim=True)
        value = torch.tensor([1])

        result = log_prob(small_normalized, value)
        # Should be close to 0 in log space (probability close to 1)
        assert result.item() <= 0 and result.item() > -1e-5

    def test_out_of_bounds_indices(self):
        """Test behavior with out of bounds indices."""
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        # This would be out of bounds for a zero-indexed array of size 3
        value = torch.tensor([10])

        # Should raise an index error
        with pytest.raises(IndexError):
            log_prob(normalized_logits, value)


class TestEntropy:
    """Test the entropy function."""

    def test_uniform_distribution(self):
        """For a uniform distribution, entropy should be log(n)."""
        n = 4
        logits = torch.ones(1, n)  # Uniform distribution when normalized
        normalized_logits = normalize_logits(logits)

        # Calculate expected entropy for uniform distribution
        probs = torch.ones(1, n) / n
        expected = -torch.sum(probs * torch.log(probs), dim=-1)

        result = entropy(normalized_logits)

        assert torch.isclose(result, expected, atol=1e-5)

    def test_deterministic_distribution(self):
        """For a nearly deterministic distribution, entropy should be close to 0."""
        # Using a more extreme logits value to create a more deterministic distribution
        logits = torch.tensor([[1000.0, 0.0, 0.0]])
        normalized_logits = normalize_logits(logits)

        # Calculate manually
        probs = torch.exp(normalized_logits)
        expected = -torch.sum(probs * torch.log(probs.clamp(min=1e-10)), dim=-1)

        result = entropy(normalized_logits)

        # Entropy should be very close to 0 for a deterministic distribution
        assert torch.isclose(result, expected, atol=1e-4)

    def test_batch_entropy(self):
        """Test entropy calculation for a batch of distributions."""
        logits = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # Uniform
                [10.0, 0.0, 0.0],  # Almost deterministic
            ]
        )
        normalized_logits = normalize_logits(logits)

        # Calculate expected entropies manually
        probs_uniform = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
        entropy_uniform = -torch.sum(probs_uniform * torch.log(probs_uniform), dim=-1)

        probs_det = torch.exp(normalized_logits[1:2])
        entropy_det = -torch.sum(probs_det * torch.log(probs_det.clamp(min=1e-10)), dim=-1)

        expected = torch.cat([entropy_uniform, entropy_det])

        result = entropy(normalized_logits)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_edge_cases(self):
        """Test edge cases for entropy function."""
        # Test with extremely low logits
        low_logits = torch.tensor([[-1000.0, -1000.0, 0.0]])
        low_normalized = normalize_logits(low_logits)

        # The calculation should not produce NaN values
        result = entropy(low_normalized)
        assert not torch.isnan(result).any()

        # Test with all equal extremely low logits
        all_low = torch.tensor([[-1000.0, -1000.0, -1000.0]])
        all_low_normalized = normalize_logits(all_low)
        result = entropy(all_low_normalized)
        assert not torch.isnan(result).any()

        # Test with empty tensor
        empty_logits = torch.ones(0, 3)
        empty_normalized = normalize_logits(empty_logits)
        result = entropy(empty_normalized)
        assert result.shape == (0,)


class TestSampleLogits:
    """Test the sample_logits function."""

    def test_sampling_shape(self, sample_logits_data):
        """Test output shapes of sample_logits."""
        single_logits = sample_logits_data["single"]
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Test with a single logits tensor
        action, logprob, ent, normalized = sample_logits([single_logits])

        # Check output shapes
        assert action.shape == torch.Size([1])  # Single batch dimension
        assert logprob.shape == (1,)
        assert ent.shape == (1,)
        assert len(normalized) == 1
        assert normalized[0].shape == single_logits.shape

        # Test with batch of logits
        action, logprob, ent, normalized = sample_logits([batch_logits])

        assert action.shape == torch.Size([batch_size])
        assert logprob.shape == (batch_size,)
        assert ent.shape == (batch_size,)
        assert len(normalized) == 1
        assert normalized[0].shape == batch_logits.shape

        # Test with multiple logits tensors
        action, logprob, ent, normalized = sample_logits([single_logits, single_logits])

        # Expected to be [num_logits * batch_size]
        assert action.shape == torch.Size([2])
        assert logprob.shape == (1,)
        assert ent.shape == (1,)
        assert len(normalized) == 2

    def test_deterministic_sampling(self, sample_logits_data):
        """Test that with deterministic logits, sampling always gives the same result."""
        deterministic_logits = sample_logits_data["deterministic"]

        # Since index 1 has the highest logit, it should always be sampled
        action, _, _, _ = sample_logits([deterministic_logits])

        assert action.item() == 1

        # Repeat sampling to ensure consistency
        for _ in range(5):
            new_action, _, _, _ = sample_logits([deterministic_logits])
            assert new_action.item() == 1

    def test_provided_actions(self, sample_logits_data):
        """Test with provided actions."""
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Create pre-specified actions
        actions = torch.tensor([0, 1, 2][:batch_size])

        # Sample with provided actions
        action, logprob, ent, normalized = sample_logits([batch_logits], action=actions)

        # Check that returned actions match provided actions
        assert torch.all(action == actions)

        # Calculate expected log probabilities manually
        normalized_logits = batch_logits - batch_logits.logsumexp(dim=-1, keepdim=True)
        expected_logprob = torch.stack([normalized_logits[i, actions[i]] for i in range(batch_size)])

        assert torch.allclose(logprob, expected_logprob)

    def test_multiple_logits_with_actions(self):
        """Test sampling from multiple logits with provided actions."""
        # Create two sets of logits
        batch_size = 2
        logits1 = torch.tensor([[1.0, 2.0, 0.0], [0.5, 1.5, 1.0]])
        logits2 = torch.tensor([[0.8, 1.2, 0.5], [1.0, 0.0, 2.0]])

        # For multiple logits, actions are flattened
        actions = torch.tensor([0, 2, 1, 0])  # [batch0,logits1], [batch1,logits1], [batch0,logits2], [batch1,logits2]

        # Sample with provided actions
        action, logprob, ent, normalized = sample_logits([logits1, logits2], action=actions)

        # Check shape of action matches expected
        assert action.shape == torch.Size([4])

        # Check values match provided actions
        assert torch.all(action == actions)

        # Normalize logits manually
        norm_logits1 = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        norm_logits2 = logits2 - logits2.logsumexp(dim=-1, keepdim=True)

        # Calculate expected log probabilities
        # For first batch (actions 0, 1), sum norm_logits1[0,0] and norm_logits2[0,1]
        # For second batch (actions 2, 0), sum norm_logits1[1,2] and norm_logits2[1,0]
        expected_logprob = torch.tensor(
            [norm_logits1[0, 0] + norm_logits2[0, 1], norm_logits1[1, 2] + norm_logits2[1, 0]]
        )

        assert torch.allclose(logprob, expected_logprob)

    def test_verbose_mode(self, sample_logits_data):
        """Test that verbose mode doesn't affect output."""
        single_logits = sample_logits_data["single"]

        # Sample with verbose=True
        action_v, logprob_v, ent_v, norm_v = sample_logits([single_logits], verbose=True)

        # Sample with verbose=False
        action, logprob, ent, norm = sample_logits([single_logits], verbose=False)

        # Output shapes should be identical regardless of verbose setting
        assert action_v.shape == action.shape
        assert logprob_v.shape == logprob.shape
        assert ent_v.shape == ent.shape
        assert len(norm_v) == len(norm)

    def test_empty_logits_list(self):
        """Test behavior with empty logits list."""
        # This should raise an IndexError since there are no logits to sample from
        with pytest.raises(IndexError):
            sample_logits([])
