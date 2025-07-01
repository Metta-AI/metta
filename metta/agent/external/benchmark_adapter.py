"""Benchmark script to measure PytorchAdapter overhead."""

import time
from types import SimpleNamespace

import torch

from metta.agent.external.pytorch_adapter import PytorchAdapter
from metta.agent.external.torch import Policy, Recurrent
from metta.agent.policy_state import PolicyState


def benchmark_adapter_overhead(num_iterations=1000, batch_size=32):
    """Measure the overhead introduced by PytorchAdapter."""

    # Create mock environment
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(nvec=[9, 10]), single_observation_space=SimpleNamespace(shape=(22, 11, 11))
    )

    # Create policies
    base_policy = Policy(env, hidden_size=512)
    recurrent_policy = Recurrent(env, policy=base_policy, input_size=512, hidden_size=512)
    adapted_policy = PytorchAdapter(recurrent_policy)

    # Create inputs
    obs = torch.randn(batch_size, 200, 3)  # Token observations
    state_dict = {"lstm_h": None, "lstm_c": None}
    state_obj = PolicyState(batch_size=batch_size)

    # Warmup
    for _ in range(10):
        _ = recurrent_policy(obs, state_dict)
        _ = adapted_policy(obs, state_obj)

    # Benchmark direct policy
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = recurrent_policy(obs, state_dict)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    direct_time = time.perf_counter() - start

    # Benchmark adapted policy
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = adapted_policy(obs, state_obj)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    adapted_time = time.perf_counter() - start

    # Calculate overhead
    overhead_ms = ((adapted_time - direct_time) / num_iterations) * 1000
    overhead_percent = ((adapted_time - direct_time) / direct_time) * 100

    print(f"Benchmark Results ({num_iterations} iterations, batch_size={batch_size}):")
    print(f"Direct policy: {direct_time:.3f}s ({direct_time / num_iterations * 1000:.3f}ms/iter)")
    print(f"Adapted policy: {adapted_time:.3f}s ({adapted_time / num_iterations * 1000:.3f}ms/iter)")
    print(f"Adapter overhead: {overhead_ms:.3f}ms/iter ({overhead_percent:.1f}%)")

    return overhead_percent


if __name__ == "__main__":
    # Run benchmarks with different batch sizes
    for batch_size in [1, 16, 32, 64]:
        print(f"\n--- Batch size: {batch_size} ---")
        benchmark_adapter_overhead(batch_size=batch_size)
