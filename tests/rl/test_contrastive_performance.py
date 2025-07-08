"""
Performance test for contrastive learning implementation.
"""

import time

import torch

from metta.rl.contrastive import ContrastiveLearning


def test_contrastive_sampling_performance():
    """Test the performance of the optimized negative sampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize contrastive learning module
    contrastive = ContrastiveLearning(hidden_size=256, gamma=0.99, temperature=0.1, logsumexp_coef=0.01, device=device)

    # Test parameters
    batch_size = 1024
    bptt_horizon = 64
    num_segments = 16
    num_rollout_negatives = 4
    num_cross_rollout_negatives = 4

    # Create test data
    current_indices = torch.randint(0, batch_size * bptt_horizon, (batch_size,), device=device)

    # Warm up
    for _ in range(10):
        _ = contrastive.sample_negative_indices(
            current_indices, num_rollout_negatives, num_cross_rollout_negatives, batch_size, bptt_horizon, num_segments
        )

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    for _ in range(100):
        negative_indices = contrastive.sample_negative_indices(
            current_indices, num_rollout_negatives, num_cross_rollout_negatives, batch_size, bptt_horizon, num_segments
        )

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"Average sampling time: {avg_time * 1000:.2f} ms")

    # Verify output shape
    expected_negatives = num_rollout_negatives + num_cross_rollout_negatives
    assert negative_indices.shape == (batch_size, expected_negatives)
    assert negative_indices.device == device


def test_contrastive_loss_performance():
    """Test the performance of the InfoNCE loss computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize contrastive learning module
    contrastive = ContrastiveLearning(hidden_size=256, gamma=0.99, temperature=0.1, logsumexp_coef=0.01, device=device)

    # Test parameters
    batch_size = 1024
    total_states = batch_size * 64  # Simulate experience buffer
    hidden_size = 256
    num_negatives = 8

    # Create test data
    lstm_states = torch.randn(batch_size, hidden_size, device=device)
    positive_indices = torch.randint(0, total_states, (batch_size,), device=device)
    negative_indices = torch.randint(0, total_states, (batch_size, num_negatives), device=device)
    all_lstm_states = torch.randn(total_states, hidden_size, device=device)

    # Warm up
    for _ in range(10):
        _ = contrastive.compute_infonce_loss(lstm_states, positive_indices, negative_indices, all_lstm_states)

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    for _ in range(100):
        loss, metrics = contrastive.compute_infonce_loss(
            lstm_states, positive_indices, negative_indices, all_lstm_states
        )

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"Average loss computation time: {avg_time * 1000:.2f} ms")

    # Verify output
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)
    assert "contrastive_total_loss" in metrics


if __name__ == "__main__":
    print("Testing contrastive learning performance...")
    test_contrastive_sampling_performance()
    test_contrastive_loss_performance()
    print("Performance tests completed!")
