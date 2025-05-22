import pytest
import torch

from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

# Global seed for reproducibility
SEED = 42


# Test fixtures
@pytest.fixture
def sample_logits_data():
    """Create sample logits of various shapes for testing."""
    # Set seed for reproducibility
    torch.manual_seed(SEED)

    batch_size = 3
    vocab_size = 5

    # Single batch, single token
    single_logits = torch.tensor([[1.0, 2.0, 0.5, -1.0, 0.0]])

    # Multiple batches, single token
    batch_logits = torch.randn(batch_size, vocab_size)

    # Create a deterministic distribution for testing
    deterministic_logits = torch.tensor([[-1000.0, 1000.0, -1000.0, -1000.0, -1000.0]])

    return {"single": single_logits, "batch": batch_logits, "deterministic": deterministic_logits}


@pytest.fixture
def benchmark_data():
    """Create benchmark data of various shapes."""
    # Set seed for reproducibility
    torch.manual_seed(SEED)

    # Small batch
    small_batch_size = 36
    small_vocab_size = 10
    small_batch = torch.randn(small_batch_size, small_vocab_size)

    # Medium batch
    medium_batch_size = 360
    medium_vocab_size = 50
    medium_batch = torch.randn(medium_batch_size, medium_vocab_size)

    # Large batch
    large_batch_size = 3600
    large_vocab_size = 1000
    large_batch = torch.randn(large_batch_size, large_vocab_size)

    # Set seed again before generating actions to ensure they're consistent
    torch.manual_seed(SEED)

    # Actions
    small_actions = torch.randint(0, small_vocab_size, (small_batch_size,))
    medium_actions = torch.randint(0, medium_vocab_size, (medium_batch_size,))
    large_actions = torch.randint(0, large_vocab_size, (large_batch_size,))

    return {
        "small_batch": small_batch,
        "medium_batch": medium_batch,
        "large_batch": large_batch,
        "small_actions": small_actions,
        "medium_actions": medium_actions,
        "large_actions": large_actions,
    }


# Test class with individual test methods
class TestSampleActions:
    """Test suite for the sample_actions function."""

    def setup_method(self):
        """Setup method called before each test method."""
        # Set seed for each test method
        torch.manual_seed(SEED)

    def test_sampling_shape(self, sample_logits_data):
        """Test output shapes of sample_actions."""
        single_logits = sample_logits_data["single"]
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Test with a single logits tensor
        action, logprob, ent, normalized = sample_actions(single_logits)

        # Check output shapes
        assert action.shape == torch.Size([1]), f"Expected action shape [1], got {action.shape}"
        assert logprob.shape == (1,), f"Expected logprob shape (1,), got {logprob.shape}"
        assert ent.shape == (1,), f"Expected entropy shape (1,), got {ent.shape}"
        assert normalized.shape == single_logits.shape, (
            f"Expected normalized shape {single_logits.shape}, got {normalized.shape}"
        )

        # Test with batch of logits
        action, logprob, ent, normalized = sample_actions(batch_logits)

        # Check batch shapes
        assert action.shape == torch.Size([batch_size]), f"Expected action shape [{batch_size}], got {action.shape}"
        assert logprob.shape == (batch_size,), f"Expected logprob shape ({batch_size},), got {logprob.shape}"
        assert ent.shape == (batch_size,), f"Expected entropy shape ({batch_size},), got {ent.shape}"
        assert normalized.shape == batch_logits.shape, (
            f"Expected normalized shape {batch_logits.shape}, got {normalized.shape}"
        )

    def test_deterministic_sampling(self, sample_logits_data):
        """Test that with deterministic logits, sampling always gives the same result."""
        deterministic_logits = sample_logits_data["deterministic"]

        # Since index 1 has the highest logit, it should always be sampled
        action, _, _, _ = sample_actions(deterministic_logits)
        assert action.item() == 1, f"Expected action 1, got {action.item()}"

        # Repeat sampling to ensure consistency
        for i in range(5):
            new_action, _, _, _ = sample_actions(deterministic_logits)
            assert new_action.item() == 1, f"Expected action 1 on iteration {i}, got {new_action.item()}"

    def test_single_element_tensor_shape(self):
        """
        Check the shape of actions when we have one agent and one batch.
        """
        # Create a tensor of shape (1, 9)
        logits = torch.randn(1, 9)

        # Call sample_actions with this tensor
        actions, logprob, logits_entropy, normalized_logits = sample_actions(logits)

        # Check shapes
        assert actions.shape == torch.Size([1]), f"Expected actions shape [1], but got {actions.shape}"
        assert logprob.shape == (1,), f"Expected logprob shape (1,), but got {logprob.shape}"
        assert logits_entropy.shape == (1,), f"Expected entropy shape (1,), but got {logits_entropy.shape}"
        assert normalized_logits.shape == logits.shape, (
            f"Expected normalized shape {logits.shape}, but got {normalized_logits.shape}"
        )


class TestEvaluateActions:
    """Test suite for the evaluate_actions function."""

    def setup_method(self):
        """Setup method called before each test method."""
        # Set seed for each test method
        torch.manual_seed(SEED)

    def test_provided_actions(self, sample_logits_data):
        """Test with provided actions."""
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Create pre-specified actions
        actions = torch.tensor([0, 1, 2][:batch_size])

        # Evaluate with provided actions
        logprob, _, normalized_logits = evaluate_actions(batch_logits, actions)

        # Calculate expected log probabilities manually
        expected_logprob = normalized_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(logprob, expected_logprob), (
            f"Function failed logprob comparison:\nExpected: {expected_logprob}\nActual: {logprob}"
        )

    def test_evaluate_shape(self, sample_logits_data):
        """Test output shapes of evaluate_actions."""
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]
        actions = torch.randint(0, batch_logits.shape[1], (batch_size,))

        # Evaluate actions
        logprob, ent, normalized = evaluate_actions(batch_logits, actions)

        # Check output shapes
        assert logprob.shape == (batch_size,), f"Expected logprob shape ({batch_size},), got {logprob.shape}"
        assert ent.shape == (batch_size,), f"Expected entropy shape ({batch_size},), got {ent.shape}"
        assert normalized.shape == batch_logits.shape, (
            f"Expected normalized shape {batch_logits.shape}, got {normalized.shape}"
        )


# Define wrapper functions for benchmarking
def run_multiple_sampling_iterations(target_func, data, iterations=10):
    """Run the sampling function multiple times to reduce variance."""
    torch.manual_seed(SEED)  # Reset seed for consistency

    for _ in range(iterations - 1):
        target_func(data)
    return target_func(data)  # Return the result of the last iteration


def run_multiple_evaluation_iterations(target_func, data, actions, iterations=10):
    """Run the evaluation function multiple times to reduce variance."""
    torch.manual_seed(SEED)  # Reset seed for consistency

    for _ in range(iterations - 1):
        target_func(data, actions)
    return target_func(data, actions)  # Return the result of the last iteration


@pytest.mark.parametrize(
    "case_name, data_key",
    [
        ("small_batch", "small_batch"),
        ("medium_batch", "medium_batch"),
        ("large_batch", "large_batch"),
    ],
)
def test_benchmark_sampling_sizes(benchmark, benchmark_data, case_name, data_key):
    """Benchmark sample_actions with different batch sizes."""
    torch.manual_seed(SEED)  # Set seed directly here
    data = benchmark_data[data_key]

    # Define a function that runs sample_actions multiple times
    def target_function():
        return run_multiple_sampling_iterations(sample_actions, data, iterations=10)

    # Use the benchmark fixture directly
    result = benchmark(target_function)

    # Validation of result
    assert result[0].shape[0] == data.shape[0], f"Expected {data.shape[0]} actions, got {result[0].shape[0]}"


@pytest.mark.parametrize(
    "case_name, data_key, action_key",
    [
        ("small_batch_with_actions", "small_batch", "small_actions"),
        ("medium_batch_with_actions", "medium_batch", "medium_actions"),
        ("large_batch_with_actions", "large_batch", "large_actions"),
    ],
)
def test_benchmark_evaluation_with_actions(benchmark, benchmark_data, case_name, data_key, action_key):
    """Benchmark evaluate_actions with provided actions."""
    torch.manual_seed(SEED)  # Set seed directly here
    data = benchmark_data[data_key]
    actions = benchmark_data[action_key]

    # Define a function that runs evaluate_actions multiple times with actions
    def target_function():
        return run_multiple_evaluation_iterations(evaluate_actions, data, actions, iterations=10)

    # Use the benchmark fixture directly
    result = benchmark(target_function)

    # Validation of result - evaluate_actions returns (logprob, entropy, normalized_logits)
    # We need to check that the function ran successfully
    assert result[0].shape[0] == actions.shape[0], f"Expected {actions.shape[0]} log probs, got {result[0].shape[0]}"


# Compatibility tests to ensure both functions work together
class TestCompatibility:
    """Test compatibility between sample_actions and evaluate_actions."""

    def test_sample_then_evaluate_consistency(self, sample_logits_data):
        """Test that sampling actions and then evaluating them gives consistent results."""
        logits = sample_logits_data["batch"]

        # Sample actions
        sampled_actions, sampled_logprob, sampled_entropy, sampled_normalized = sample_actions(logits)

        # Evaluate the same actions
        eval_logprob, eval_entropy, eval_normalized = evaluate_actions(logits, sampled_actions)

        # Results should be identical
        assert torch.allclose(sampled_logprob, eval_logprob), "Log probabilities should match"
        assert torch.allclose(sampled_entropy, eval_entropy), "Entropies should match"
        assert torch.allclose(sampled_normalized, eval_normalized), "Normalized logits should match"
