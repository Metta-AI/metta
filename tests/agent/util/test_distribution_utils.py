import time
from typing import Callable, Optional, Tuple

import pytest
import torch

from metta.agent.util.distribution_utils import (
    sample_logits_main,
    sample_logits_new,
    sample_logits_patched,
)


# Create test fixtures that can be reused for all three implementations
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


# Base test class that can be subclassed for each implementation
class BaseTestSampleLogits:
    """Base test class for sample_logits functions."""

    # Override this in subclasses
    sample_logits_func: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None

    def test_sampling_shape(self, sample_logits_data):
        """Test output shapes of sample_logits."""
        assert self.sample_logits_func is not None, "sample_logits_func must be defined in subclass"

        single_logits = sample_logits_data["single"]
        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Test with a single logits tensor
        action, logprob, ent, normalized = self.sample_logits_func([single_logits])

        # Check output shapes - accept either [1] or [1, 1] shape for actions
        assert action.shape == torch.Size([1]) or action.shape == torch.Size([1, 1])
        assert logprob.shape == (1,)
        assert ent.shape == (1,)
        assert len(normalized) == 1
        assert normalized[0].shape == single_logits.shape

        # Test with batch of logits
        action, logprob, ent, normalized = self.sample_logits_func([batch_logits])

        # Accept either [batch_size] or [batch_size, 1] for batch actions
        assert action.shape == torch.Size([batch_size]) or action.shape == torch.Size([batch_size, 1])
        assert logprob.shape == (batch_size,)
        assert ent.shape == (batch_size,)
        assert len(normalized) == 1
        assert normalized[0].shape == batch_logits.shape

        # Test with multiple logits tensors
        action, logprob, ent, normalized = self.sample_logits_func([single_logits, single_logits])

        # Expected action shape for two single-batch tensors
        assert action.shape == torch.Size([1, 2]) or action.shape == torch.Size([2])
        assert logprob.shape == (1,)
        assert ent.shape == (1,)
        assert len(normalized) == 2

    def test_deterministic_sampling(self, sample_logits_data):
        """Test that with deterministic logits, sampling always gives the same result."""
        assert self.sample_logits_func is not None, "sample_logits_func must be defined in subclass"

        deterministic_logits = sample_logits_data["deterministic"]

        # Since index 1 has the highest logit, it should always be sampled
        action, _, _, _ = self.sample_logits_func([deterministic_logits])

        assert action.item() == 1

        # Repeat sampling to ensure consistency
        for _ in range(5):
            new_action, _, _, _ = self.sample_logits_func([deterministic_logits])
            assert new_action.item() == 1

    def test_provided_actions(self, sample_logits_data):
        """Test with provided actions."""
        assert self.sample_logits_func is not None, "sample_logits_func must be defined in subclass"

        batch_logits = sample_logits_data["batch"]
        batch_size = batch_logits.shape[0]

        # Create pre-specified actions
        actions = torch.tensor([0, 1, 2][:batch_size])

        # Sample with provided actions
        action, logprob, ent, normalized = self.sample_logits_func([batch_logits], action=actions)

        # Flatten output if needed
        assert torch.equal(action.view(-1), actions)

        # Calculate expected log probabilities manually
        normalized_logits = batch_logits - batch_logits.logsumexp(dim=-1, keepdim=True)
        expected_logprob = torch.stack([normalized_logits[i, actions[i]] for i in range(batch_size)])

        assert torch.allclose(logprob, expected_logprob), (
            f"{self.sample_logits_func.__name__} failed logprob comparison:\n"
            f"Expected: {expected_logprob}\n"
            f"Actual: {logprob}\n"
        )

    def test_multiple_logits_with_actions(self):
        """Test sampling from multiple logits with provided actions."""
        assert self.sample_logits_func is not None, "sample_logits_func must be defined in subclass"

        # Create two sets of logits
        _batch_size = 2
        logits1 = torch.tensor([[1.0, 2.0, 0.0], [0.5, 1.5, 1.0]])
        logits2 = torch.tensor([[0.8, 1.2, 0.5], [1.0, 0.0, 2.0]])

        # For multiple logits, actions are flattened
        actions = torch.tensor([0, 2, 1, 0])  # [batch0,logits1], [batch1,logits1], [batch0,logits2], [batch1,logits2]

        # Sample with provided actions
        action, logprob, ent, normalized = self.sample_logits_func([logits1, logits2], action=actions)

        # Handle potential action reshaping during sampling
        # Convert to flat format for comparison regardless of internal representation
        if action.dim() == 2 and action.shape == torch.Size([2, 2]):
            # If action is [2, 2], flatten it
            action_flat = action.flatten()
        else:
            # If action is already flattened
            action_flat = action

        # Check values match provided actions in the appropriate order
        assert torch.all(action_flat == actions), f"Expected {actions}, got flattened {action_flat}"

        # Normalize logits manually
        norm_logits1 = logits1 - logits1.logsumexp(dim=-1, keepdim=True)
        norm_logits2 = logits2 - logits2.logsumexp(dim=-1, keepdim=True)

        # Calculate expected log probabilities
        # For first batch (actions 0, 1), sum norm_logits1[0,0] and norm_logits2[0,1]
        # For second batch (actions 2, 0), sum norm_logits1[1,2] and norm_logits2[1,0]
        expected_logprob = torch.tensor(
            [norm_logits1[0, 0] + norm_logits2[0, 1], norm_logits1[1, 2] + norm_logits2[1, 0]]
        )

        assert torch.allclose(logprob, expected_logprob), (
            f"{self.sample_logits_func.__name__} failed logprob comparison:\n"
            f"Expected: {expected_logprob}\n"
            f"Actual: {logprob}\n"
        )

    def test_single_element_list_shape(self):
        """Check the shape of actions when we have one agent and one batch."""
        assert self.sample_logits_func is not None, "sample_logits_func must be defined in subclass"

        # Create a list with a single tensor of shape (1, 9)
        logits_list = [torch.randn(1, 9)]

        # Call sample_logits with this list
        actions, logprob, logits_entropy, normalized_logits = self.sample_logits_func(logits_list)

        # Print actual shape for debugging
        print(f"Expected shape: torch.Size([1, 1]), Got shape: {actions.shape}")

        # This assertion will fail because actions has shape (1) instead of (1, 1)
        assert actions.shape == torch.Size([1, 1]), (
            f"{self.sample_logits_func.__name__} failed logprob comparison:\n"
            f"Expected actions shape (1, 1), but got {actions.shape}"
        )


# Subclass for sample_logits_main
class TestSampleLogitsMain(BaseTestSampleLogits):
    """Test the sample_logits_main function."""

    sample_logits_func = staticmethod(sample_logits_main)


# Subclass for sample_logits_patched
class TestSampleLogitsPatched(BaseTestSampleLogits):
    """Test the sample_logits_patched function."""

    sample_logits_func = staticmethod(sample_logits_patched)


# Subclass for sample_logits_new
class TestSampleLogitsNew(BaseTestSampleLogits):
    """Test the sample_logits_new function."""

    sample_logits_func = staticmethod(sample_logits_new)


@pytest.fixture
def benchmark_data():
    """Create benchmark data of various shapes."""
    # Small batch
    small_batch_size = 4
    small_vocab_size = 10
    small_batch = torch.randn(small_batch_size, small_vocab_size)

    # Medium batch
    medium_batch_size = 32
    medium_vocab_size = 50
    medium_batch = torch.randn(medium_batch_size, medium_vocab_size)

    # Large batch
    large_batch_size = 128
    large_vocab_size = 1000
    large_batch = torch.randn(large_batch_size, large_vocab_size)

    # Multiple logits
    multi_small = [torch.randn(small_batch_size, small_vocab_size) for _ in range(3)]
    multi_medium = [torch.randn(medium_batch_size, medium_vocab_size) for _ in range(3)]

    # Actions
    small_actions = torch.randint(0, small_vocab_size, (small_batch_size,))
    medium_actions = torch.randint(0, medium_vocab_size, (medium_batch_size,))
    multi_small_actions = torch.randint(0, small_vocab_size, (small_batch_size * 3,))

    return {
        "small_batch": small_batch,
        "medium_batch": medium_batch,
        "large_batch": large_batch,
        "multi_small": multi_small,
        "multi_medium": multi_medium,
        "small_actions": small_actions,
        "medium_actions": medium_actions,
        "multi_small_actions": multi_small_actions,
    }


# Functions to benchmark all three implementations
def benchmark_implementation(func, data, action=None, num_runs=100):
    """Benchmark a specific implementation."""
    start_time = time.time()
    for _ in range(num_runs):
        result = func(data, action=action)
    total_time = time.time() - start_time
    return total_time / num_runs, result


def run_benchmark_all_implementations(data, action=None, num_runs=100):
    """Run benchmark and return comparative results."""
    funcs = {
        "sample_logits_patched": sample_logits_patched,
        "sample_logits_new": sample_logits_new,
    }

    results = {}
    for name, func in funcs.items():
        avg_time, _ = benchmark_implementation(func, data, action, num_runs)
        results[name] = avg_time

    return results


@pytest.mark.verbose
@pytest.mark.benchmark
def test_benchmark_comparison(benchmark_data):
    """Compare performance of all implementations."""
    print("\n=== BENCHMARK COMPARISON ===")

    # Test cases
    test_cases = [
        ("Small Batch", [benchmark_data["small_batch"]], None),
        ("Medium Batch", [benchmark_data["medium_batch"]], None),
        ("Large Batch", [benchmark_data["large_batch"]], None),
        ("Multiple Small Logits", benchmark_data["multi_small"], None),
        ("Small Batch with Actions", [benchmark_data["small_batch"]], benchmark_data["small_actions"]),
        ("Multiple Small Logits with Actions", benchmark_data["multi_small"], benchmark_data["multi_small_actions"]),
    ]

    # Run benchmarks for each test case
    for name, data, action in test_cases:
        print(f"\n--- {name} ---")
        results = run_benchmark_all_implementations(data, action, num_runs=100)

        # Find the fastest implementation to use as baseline
        fastest = min(results.values())

        # Print results
        for impl, time_taken in results.items():
            relative = time_taken / fastest
            print(f"{impl}: {time_taken:.6f} seconds ({relative:.2f}x)")


def test_sample_action_consistency(benchmark_data):
    """Verify that all sample_logits implementations produce consistent actions."""
    funcs = {
        "sample_logits_patched": sample_logits_patched,
        "sample_logits_new": sample_logits_new,
    }

    test_cases = [
        ("Small Batch", [benchmark_data["small_batch"]], None),
        ("Multiple Small Logits", benchmark_data["multi_small"], None),
        ("Small Batch with Actions", [benchmark_data["small_batch"]], benchmark_data["small_actions"]),
        ("Multiple Small Logits with Actions", benchmark_data["multi_small"], benchmark_data["multi_small_actions"]),
    ]

    for case_name, data, action in test_cases:
        results = {}
        for name, func in funcs.items():
            torch.manual_seed(42)  # 🔥 reset RNG before every sampling
            act, _, _, _ = func(data, action=action)
            results[name] = act.reshape(-1)  # flatten actions if needed

        # Compare each pair
        names = list(results.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                r1, r2 = results[name1], results[name2]

                assert torch.equal(r1, r2), (
                    f"Action mismatch between {name1} and {name2} for {case_name}:\n"
                    f"{name1} actions: {r1}\n"
                    f"{name2} actions: {r2}"
                )
