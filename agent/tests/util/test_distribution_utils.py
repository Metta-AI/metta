import pytest
import torch

from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

# Global seed for reproducibility
SEED = 42


@pytest.fixture
def sample_logits_data():
    """Create sample logits of various shapes for testing."""
    torch.manual_seed(SEED)

    batch_size = 3
    vocab_size = 5

    single_logits = torch.tensor([[1.0, 2.0, 0.5, -1.0, 0.0]], dtype=torch.float32)
    batch_logits = torch.randn(batch_size, vocab_size)
    deterministic_logits = torch.tensor([[-1000.0, 1000.0, -1000.0, -1000.0, -1000.0]], dtype=torch.float32)

    return {
        "single": single_logits,
        "batch": batch_logits,
        "deterministic": deterministic_logits,
    }


@pytest.fixture
def benchmark_data():
    """Create large-scale test data for evaluating scalability and shape handling."""
    torch.manual_seed(SEED)

    small_batch = torch.randn(36, 10)
    medium_batch = torch.randn(360, 50)
    large_batch = torch.randn(3600, 1000)

    torch.manual_seed(SEED)
    small_actions = torch.randint(0, 10, (36,))
    medium_actions = torch.randint(0, 50, (360,))
    large_actions = torch.randint(0, 1000, (3600,))

    return {
        "small_batch": small_batch,
        "medium_batch": medium_batch,
        "large_batch": large_batch,
        "small_actions": small_actions,
        "medium_actions": medium_actions,
        "large_actions": large_actions,
    }


class TestSampleActions:
    def setup_method(self):
        torch.manual_seed(SEED)

    def test_sampling_shape(self, sample_logits_data):
        single = sample_logits_data["single"]
        batch = sample_logits_data["batch"]
        batch_size = batch.shape[0]

        action, logprob, ent, norm = sample_actions(single)
        assert action.shape == (1,)
        assert logprob.shape == (1,)
        assert ent.shape == (1,)
        assert norm.shape == single.shape

        action, logprob, ent, norm = sample_actions(batch)
        assert action.shape == (batch_size,)
        assert logprob.shape == (batch_size,)
        assert ent.shape == (batch_size,)
        assert norm.shape == batch.shape

    def test_deterministic_sampling(self, sample_logits_data):
        logits = sample_logits_data["deterministic"]

        for i in range(6):
            action, _, _, _ = sample_actions(logits)
            assert action.item() == 1, f"Deterministic sampling failed on iteration {i}"

    def test_single_element_tensor_shape(self):
        logits = torch.randn(1, 9)
        actions, logprob, entropy, norm = sample_actions(logits)
        assert actions.shape == (1,)
        assert logprob.shape == (1,)
        assert entropy.shape == (1,)
        assert norm.shape == logits.shape


class TestEvaluateActions:
    def setup_method(self):
        torch.manual_seed(SEED)

    def test_provided_actions(self, sample_logits_data):
        logits = sample_logits_data["batch"]
        actions = torch.tensor([0, 1, 2][: logits.shape[0]], dtype=torch.long)
        logprob, _, norm = evaluate_actions(logits, actions)
        expected = norm.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(logprob, expected)

    def test_evaluate_shape(self, sample_logits_data):
        logits = sample_logits_data["batch"]
        batch_size, vocab_size = logits.shape
        actions = torch.randint(0, vocab_size, (batch_size,))
        logprob, ent, norm = evaluate_actions(logits, actions)
        assert logprob.shape == (batch_size,)
        assert ent.shape == (batch_size,)
        assert norm.shape == logits.shape


def run_multiple_sampling_iterations(func, data, iterations=10):
    torch.manual_seed(SEED)
    for _ in range(iterations - 1):
        func(data)
    return func(data)


def run_multiple_evaluation_iterations(func, data, actions, iterations=10):
    torch.manual_seed(SEED)
    for _ in range(iterations - 1):
        func(data, actions)
    return func(data, actions)


@pytest.mark.parametrize(
    "data_key",
    ["small_batch", "medium_batch", "large_batch"],
)
def test_sampling_output_shapes(benchmark_data, data_key):
    data = benchmark_data[data_key]
    actions, logprob, entropy, norm = run_multiple_sampling_iterations(sample_actions, data)

    assert actions.shape[0] == data.shape[0]
    assert logprob.shape[0] == data.shape[0]
    assert entropy.shape[0] == data.shape[0]
    assert norm.shape == data.shape


@pytest.mark.parametrize(
    "data_key,action_key",
    [
        ("small_batch", "small_actions"),
        ("medium_batch", "medium_actions"),
        ("large_batch", "large_actions"),
    ],
)
def test_evaluation_output_shapes(benchmark_data, data_key, action_key):
    data = benchmark_data[data_key]
    actions = benchmark_data[action_key]
    logprob, entropy, norm = run_multiple_evaluation_iterations(evaluate_actions, data, actions)

    assert logprob.shape[0] == data.shape[0]
    assert entropy.shape[0] == data.shape[0]
    assert norm.shape == data.shape


class TestCompatibility:
    def test_sample_then_evaluate_consistency(self, sample_logits_data):
        logits = sample_logits_data["batch"]
        act, lp, ent, norm = sample_actions(logits)
        eval_lp, eval_ent, eval_norm = evaluate_actions(logits, act)

        assert torch.allclose(lp, eval_lp), "Logprobs mismatch"
        assert torch.allclose(ent, eval_ent), "Entropy mismatch"
        assert torch.allclose(norm, eval_norm), "Normalized logits mismatch"
