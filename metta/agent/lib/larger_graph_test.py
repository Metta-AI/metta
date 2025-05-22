"""
Test with larger neural network graphs to see algorithm scaling.
"""

from performance_comparison import GraphPerformanceTest


def create_large_nn_graph(graph, size=20):
    """Create a larger neural network graph."""
    modules = []

    # Input processing pipeline
    modules.extend([f"obs_norm_{i}" for i in range(3)])
    modules.extend([f"embed_{i}" for i in range(2)])

    # Main network backbone
    modules.extend([f"encoder_{i}" for i in range(4)])
    modules.extend([f"lstm_{i}" for i in range(2)])

    # Multi-head outputs
    modules.extend([f"policy_head_{i}" for i in range(3)])
    modules.extend([f"value_head_{i}" for i in range(2)])
    modules.extend([f"auxiliary_head_{i}" for i in range(4)])

    # Take only the first 'size' modules
    modules = modules[:size]

    for module in modules:
        graph.add_module(module)

    # Create more complex connection patterns
    connections = []

    # Sequential backbone
    for i in range(len(modules) - 1):
        if "obs_norm" in modules[i] and "obs_norm" in modules[i + 1]:
            connections.append((modules[i], modules[i + 1]))
        elif "embed" in modules[i] and "embed" in modules[i + 1]:
            connections.append((modules[i], modules[i + 1]))
        elif "encoder" in modules[i] and "encoder" in modules[i + 1]:
            connections.append((modules[i], modules[i + 1]))
        elif "lstm" in modules[i] and "lstm" in modules[i + 1]:
            connections.append((modules[i], modules[i + 1]))

    # Cross-connections between stages
    obs_norms = [m for m in modules if "obs_norm" in m]
    embeds = [m for m in modules if "embed" in m]
    encoders = [m for m in modules if "encoder" in m]
    lstms = [m for m in modules if "lstm" in m]
    heads = [m for m in modules if "head" in m]

    if obs_norms and embeds:
        connections.append((obs_norms[-1], embeds[0]))
    if embeds and encoders:
        connections.append((embeds[-1], encoders[0]))
    if encoders and lstms:
        connections.append((encoders[-1], lstms[0]))

    # Connect last LSTM to all heads
    if lstms and heads:
        for head in heads:
            connections.append((lstms[-1], head))

    # Add some fan-out connections
    if len(encoders) >= 2 and heads:
        for head in heads[:2]:
            connections.append((encoders[-2], head))

    for source, target in connections:
        graph.connect(source, target)

    return len(modules), len(connections)


def test_scaling():
    """Test algorithm performance at different graph sizes."""
    sizes = [5, 10, 15, 20]

    print("=== Algorithm Scaling Test ===\n")

    for size in sizes:
        print(f"Testing graph size: {size} modules")

        graph = GraphPerformanceTest()
        num_modules, num_connections = create_large_nn_graph(graph, size)

        print(f"  Actual size: {num_modules} modules, {num_connections} edges")

        # Quick benchmark
        results = graph.benchmark_algorithms(1000)  # Fewer iterations for larger graphs

        for algorithm, metrics in results.items():
            print(f"  {algorithm}: {metrics['avg_time_us']:.2f} μs")

        fastest = min(results.items(), key=lambda x: x[1]["avg_time_us"])
        print(f"  🏆 Fastest: {fastest[0]}")
        print()


if __name__ == "__main__":
    test_scaling()
